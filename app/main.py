from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data")).resolve()
CONFIG_FILE = ROOT_DIR / "config.json"
LIBRARIES_DIR = DATA_DIR / "libraries"
MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_EXTS = {".skp", ".3ds", ".max", ".obj", ".fbx", ".blend", ".dae", ".dwg", ".dxf", ".rvt", ".stl", ".glb", ".gltf"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z"}
EXCLUDED_DIR_NAMES = {"texture", "textures", "tex", "map", "maps", "material", "materials", "材質", "貼圖", "mapsources", "assets"}
TEXTURE_NAME_TOKENS = {"diffuse", "albedo", "basecolor", "base_color", "normal", "bump", "height", "displacement", "roughness", "rough", "metallic", "metalness", "opacity", "alpha", "mask", "specular", "gloss", "ambientocclusion", "ambient_occlusion", "emissive"}
UNCATEGORIZED = "未分類"
ALL_LIBRARIES = "__all__"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)


def _slug(text: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip())
    return (safe[:40] or "library") + "_" + uuid.uuid4().hex[:8]


def load_config() -> dict[str, Any]:
    defaults = {
        "search_limit": 24, "port": 8090, "device": "auto",
        "gpu_batch_size": 32, "cpu_batch_size": 4, "checkpoint_every_batches": 5,
        "libraries": [], "active_library_id": "",
    }
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig")) if CONFIG_FILE.exists() else {}
    except Exception:
        raw = {}
    cfg = {**defaults, **raw}
    # Migrate legacy single-library config.
    if not cfg.get("libraries"):
        legacy = str(raw.get("library_dir") or os.getenv("LIBRARY_DIR") or "").strip()
        if legacy:
            lib_id = "main"
            cfg["libraries"] = [{"id": lib_id, "name": "主要模型庫", "path": legacy}]
            cfg["active_library_id"] = lib_id
    if cfg.get("libraries") and not cfg.get("active_library_id"):
        cfg["active_library_id"] = cfg["libraries"][0]["id"]
    return cfg


def save_config() -> None:
    # Keep legacy library_dir for launcher compatibility.
    active = next((x for x in _config["libraries"] if x["id"] == _config.get("active_library_id")), None)
    payload = dict(_config)
    payload["library_dir"] = active["path"] if active else ""
    CONFIG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_config = load_config()
save_config()


def _library_by_id(lib_id: str | None = None) -> dict[str, str]:
    target = lib_id or _config.get("active_library_id")
    lib = next((x for x in _config["libraries"] if x["id"] == target), None)
    if not lib:
        raise HTTPException(status_code=404, detail="找不到模型庫設定")
    return lib


def _paths_for(lib_id: str) -> dict[str, Path]:
    base = LIBRARIES_DIR / lib_id
    index = base / "index"
    checkpoint = base / "checkpoint"
    index.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    return {
        "base": base, "index_dir": index, "index": index / "index.npz",
        "meta": index / "metadata.json", "manifest": index / "manifest.json",
        "checkpoint_dir": checkpoint, "cp_index": checkpoint / "partial.npz",
        "cp_meta": checkpoint / "metadata.json", "cp_manifest": checkpoint / "manifest.json",
        "cp_job": checkpoint / "job.json",
    }


def _migrate_legacy_data() -> None:
    if not _config.get("libraries"):
        return
    lib_id = _config["active_library_id"]
    dst = _paths_for(lib_id)
    legacy_index = DATA_DIR / "index"
    legacy_cp = DATA_DIR / "checkpoint"
    if legacy_index.exists() and not dst["index"].exists():
        for src_name, key in (("index.npz", "index"), ("metadata.json", "meta"), ("manifest.json", "manifest")):
            src = legacy_index / src_name
            if src.exists(): shutil.copy2(src, dst[key])
    if legacy_cp.exists() and not dst["cp_job"].exists():
        for src_name, key in (("partial.npz", "cp_index"), ("metadata.json", "cp_meta"), ("manifest.json", "cp_manifest"), ("job.json", "cp_job")):
            src = legacy_cp / src_name
            if src.exists(): shutil.copy2(src, dst[key])


_migrate_legacy_data()
app = FastAPI(title="3D 模型以圖搜圖－正式免安裝版 v1.6")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

_device_pref = str(os.getenv("DEVICE_MODE") or _config.get("device", "auto")).lower()
_device = "cpu" if _device_pref == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
_model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
_model = _model.to(_device).eval()
_gpu_name = torch.cuda.get_device_name(0) if _device == "cuda" else None
_batch_size = int(_config.get("gpu_batch_size", 32) if _device == "cuda" else _config.get("cpu_batch_size", 4))
_checkpoint_every = max(1, int(_config.get("checkpoint_every_batches", 5)))
_index_lock = threading.Lock()
_vectors: np.ndarray | None = None
_metadata: list[dict[str, Any]] = []
_manifest: dict[str, dict[str, Any]] = {}
_indexing = False
_pause_event = threading.Event()
_stop_event = threading.Event()
_status: dict[str, Any] = {"message": "尚未建立索引", "done": 0, "total": 0, "mode": "idle", "paused": False}


def active_library() -> dict[str, str]: return _library_by_id()
def library_root(lib_id: str | None = None) -> Path: return Path(_library_by_id(lib_id)["path"]).resolve()
def active_paths() -> dict[str, Path]: return _paths_for(active_library()["id"])



def _category_info_for_rel(rel: str) -> tuple[str, str, str]:
    """Return category, subcategory and complete category path.

    The image's parent path is treated as the category hierarchy. The first
    level remains compatible with older indexes, while deeper levels are kept
    for display and future filtering.
    """
    parent_parts = Path(rel).parent.parts
    if not parent_parts or parent_parts == (".",):
        return UNCATEGORIZED, "", UNCATEGORIZED
    category = parent_parts[0]
    subcategory = parent_parts[1] if len(parent_parts) > 1 else ""
    category_path = "/".join(parent_parts)
    return category, subcategory, category_path


def _category_for_rel(rel: str) -> str:
    return _category_info_for_rel(rel)[0]


def _looks_like_texture_name(path: Path) -> bool:
    stem = path.stem.lower().replace("-", "_").replace(" ", "_")
    tokens = {t for t in re.split(r"[^a-z0-9\u4e00-\u9fff]+", stem) if t}
    if tokens & TEXTURE_NAME_TOKENS:
        return True
    return any(token in stem for token in TEXTURE_NAME_TOKENS if len(token) >= 5)


def _is_extracted_payload_dir(directory: Path) -> bool:
    """Detect a model's extracted payload folder.

    Example:
      沙發001.jpg
      沙發001.zip
      沙發001/       <- skip recursively
    """
    parent = directory.parent
    stem = directory.name.lower()
    try:
        siblings = list(parent.iterdir())
    except OSError:
        return False

    for item in siblings:
        if not item.is_file():
            continue
        if item.stem.lower() != stem:
            continue
        if item.suffix.lower() in IMAGE_EXTS | ARCHIVE_EXTS | MODEL_EXTS:
            return True
    return False


def _preview_marker_for_image(image_path: Path) -> bool:
    """Return True when an image looks like a model preview rather than texture."""
    stem = image_path.stem.lower()
    parent = image_path.parent
    try:
        siblings = list(parent.iterdir())
    except OSError:
        return False

    # Exact same-name archive, extracted directory, or model file.
    for item in siblings:
        if item.stem.lower() == stem:
            if item.is_dir() or (item.is_file() and item.suffix.lower() in ARCHIVE_EXTS | MODEL_EXTS):
                return True

    # Some collections use one preview image for several package/model files.
    package_files = [
        p for p in siblings
        if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS | MODEL_EXTS
    ]
    if package_files:
        # Prefer name similarity, but still accept ordinary images in a package folder.
        if any(p.stem.lower().startswith(stem) or stem.startswith(p.stem.lower()) for p in package_files):
            return True
        if not _looks_like_texture_name(image_path):
            return True

    return False


def _smart_scan_all(root: Path) -> tuple[list[Path], dict[str, int]]:
    """Scan arbitrary category depth while excluding extracted model textures."""
    selected: list[Path] = []
    stats = {
        "preview_images": 0,
        "excluded_texture_names": 0,
        "excluded_payload_dirs": 0,
        "excluded_texture_dirs": 0,
        "unmarked_images": 0,
    }

    for current_str, dirs, files in os.walk(root, topdown=True):
        current = Path(current_str)

        # Prune texture/material folders and same-name extracted model folders.
        kept_dirs = []
        for dirname in dirs:
            child = current / dirname
            if dirname.lower() in EXCLUDED_DIR_NAMES:
                stats["excluded_texture_dirs"] += 1
                continue
            if _is_extracted_payload_dir(child):
                stats["excluded_payload_dirs"] += 1
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        image_files = [
            current / name for name in files
            if Path(name).suffix.lower() in IMAGE_EXTS
        ]
        if not image_files:
            continue

        for image_path in image_files:
            if _looks_like_texture_name(image_path):
                stats["excluded_texture_names"] += 1
                continue
            if _preview_marker_for_image(image_path):
                selected.append(image_path)
                stats["preview_images"] += 1
            else:
                # Compatibility fallback: images at a leaf category folder are
                # retained only when the folder has no subdirectories and the
                # image does not resemble a material map.
                if not dirs:
                    selected.append(image_path)
                    stats["preview_images"] += 1
                else:
                    stats["unmarked_images"] += 1

    return sorted(set(selected)), stats


def _encode_image(image: Image.Image) -> np.ndarray:
    tensor = _preprocess(image.convert("RGB")).unsqueeze(0).to(_device)
    with torch.inference_mode():
        vec = _model.encode_image(tensor); vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.detach().cpu().numpy().astype("float32")[0]


def _encode_paths(paths: list[Path]) -> list[np.ndarray | None]:
    tensors, valid = [], []
    results: list[np.ndarray | None] = [None] * len(paths)
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as img: tensors.append(_preprocess(img.convert("RGB"))); valid.append(i)
        except Exception as exc: print(f"Skip {path}: {exc}")
    if not tensors: return results
    batch = torch.stack(tensors).to(_device, non_blocking=True)
    with torch.inference_mode():
        vecs = _model.encode_image(batch); vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    arr = vecs.detach().cpu().numpy().astype("float32")
    for oi, si in enumerate(valid): results[si] = arr[oi]
    return results


def _find_model_for_image(image_path: Path) -> Path | None:
    # Exact same-name archive/package has priority.
    for ext in ARCHIVE_EXTS:
        candidate = image_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    # Exact same-name model file.
    for ext in MODEL_EXTS:
        candidate = image_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    # Same-name extracted folder: locate a model file inside without scanning
    # unrelated category folders. If none exists, return the folder itself so
    # "open file location" can still select it.
    extracted = image_path.parent / image_path.stem
    if extracted.is_dir():
        try:
            models = [
                p for p in extracted.rglob("*")
                if p.is_file() and p.suffix.lower() in MODEL_EXTS
            ]
            if models:
                return sorted(models, key=lambda p: (len(p.parts), str(p).lower()))[0]
        except OSError:
            pass
        return extracted

    try:
        candidates = [
            p for p in image_path.parent.iterdir()
            if p.is_file() and p.suffix.lower() in (MODEL_EXTS | ARCHIVE_EXTS)
        ]
    except Exception:
        return None
    if not candidates:
        return None
    stem = image_path.stem.lower()
    starts = [
        p for p in candidates
        if p.stem.lower().startswith(stem) or stem.startswith(p.stem.lower())
    ]
    return sorted(starts or candidates)[0]


def _scan_images(category: str = "全部") -> list[Path]:
    root = library_root()
    if not root.exists():
        return []

    images, stats = _smart_scan_all(root)
    _status["scan_stats"] = stats

    if category == "全部":
        return images
    if category == UNCATEGORIZED:
        return [p for p in images if _category_for_rel(str(p.relative_to(root))) == UNCATEGORIZED]
    return [
        p for p in images
        if _category_for_rel(str(p.relative_to(root))) == category
    ]

def _fingerprint(path: Path) -> dict[str, Any]:
    s = path.stat(); return {"size": s.st_size, "mtime_ns": s.st_mtime_ns}


def _atomic_save(vectors: np.ndarray, metadata: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> None:
    p = active_paths(); tmp = p["index_dir"] / "index.new.npz"; tm = p["index_dir"] / "metadata.new.json"; tf = p["index_dir"] / "manifest.new.json"
    np.savez_compressed(tmp, vectors=vectors); tm.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"); tf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for src, dst in ((tmp,p["index"]),(tm,p["meta"]),(tf,p["manifest"])):
        if dst.exists(): shutil.copy2(dst, dst.with_suffix(dst.suffix+".bak"))
        os.replace(src,dst)


def _clear_checkpoint(lib_id: str | None = None) -> None:
    p = _paths_for(lib_id or active_library()["id"])
    for k in ("cp_index","cp_meta","cp_manifest","cp_job"):
        try: p[k].unlink(missing_ok=True)
        except Exception: pass


def _checkpoint_info() -> dict[str, Any] | None:
    p=active_paths()
    if not all(p[k].exists() for k in ("cp_index","cp_meta","cp_manifest","cp_job")): return None
    try: return json.loads(p["cp_job"].read_text(encoding="utf-8"))
    except Exception: return None


def _save_checkpoint(vectors, metadata, manifest, pending, done, total, full, category):
    p=active_paths(); dim=vectors[0].shape[0] if vectors else 512; matrix=np.vstack(vectors).astype("float32") if vectors else np.empty((0,dim),dtype="float32")
    tmp=p["checkpoint_dir"]/"partial.new.npz"; tm=p["checkpoint_dir"]/"metadata.new.json"; tf=p["checkpoint_dir"]/"manifest.new.json"; tj=p["checkpoint_dir"]/"job.new.json"
    np.savez_compressed(tmp,vectors=matrix); tm.write_text(json.dumps(metadata,ensure_ascii=False),encoding="utf-8"); tf.write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
    tj.write_text(json.dumps({"version":2,"library_id":active_library()["id"],"full":full,"category":category,"pending":pending,"done":done,"total":total},ensure_ascii=False,indent=2),encoding="utf-8")
    for src,dst in ((tmp,p["cp_index"]),(tm,p["cp_meta"]),(tf,p["cp_manifest"]),(tj,p["cp_job"])): os.replace(src,dst)


def _load_checkpoint():
    p=active_paths(); job=_checkpoint_info()
    if not job: raise RuntimeError("沒有可繼續的索引進度")
    matrix=np.load(p["cp_index"])["vectors"].astype("float32")
    return [matrix[i] for i in range(len(matrix))], json.loads(p["cp_meta"].read_text(encoding="utf-8")), json.loads(p["cp_manifest"].read_text(encoding="utf-8")), job


def _load_index() -> None:
    global _vectors,_metadata,_manifest,_status
    p=active_paths()
    try:
        if p["index"].exists() and p["meta"].exists():
            _vectors=np.load(p["index"])["vectors"].astype("float32"); _metadata=json.loads(p["meta"].read_text(encoding="utf-8")); _manifest=json.loads(p["manifest"].read_text(encoding="utf-8")) if p["manifest"].exists() else {}
            for x in _metadata:
                cat, subcat, cat_path = _category_info_for_rel(x.get("image", ""))
                x.setdefault("category", cat)
                x.setdefault("subcategory", subcat)
                x.setdefault("category_path", cat_path)
            _status={"message":f"索引已載入，共 {len(_metadata)} 張圖片","done":len(_metadata),"total":len(_metadata),"mode":"idle","paused":False}
        else:
            _vectors=None; _metadata=[]; _manifest={}; _status={"message":"尚未建立索引","done":0,"total":0,"mode":"idle","paused":False}
    except Exception as exc:
        _vectors=None;_metadata=[];_manifest={};_status={"message":f"索引載入失敗：{exc}","done":0,"total":0,"mode":"error","paused":False}


def _build_index_worker(full=False, category="全部", resume=False):
    global _vectors,_metadata,_manifest,_indexing,_status
    try:
        _pause_event.clear();_stop_event.clear(); root=library_root(); mode_text="完整重建" if full else "增量更新"; mode="full" if full else "incremental"
        if resume:
            keep_vectors,keep_meta,new_manifest,job=_load_checkpoint(); full=bool(job.get("full")); category=str(job.get("category","全部")); mode_text="完整重建" if full else "增量更新"; mode="full" if full else "incremental"; pending=list(job.get("pending",[])); total=int(job.get("total",len(pending))); done=int(job.get("done",0)); changed=[(r,root/Path(r)) for r in pending if (root/Path(r)).exists()]
        else:
            _clear_checkpoint(); images=_scan_images(category); scope={str(p.relative_to(root)).replace("\\","/"):p for p in images}; old={m.get("image"):(i,m) for i,m in enumerate(_metadata)}; keep_vectors=[];keep_meta=[];new_manifest={};changed=[]
            for rel,(idx,meta) in old.items():
                if category!="全部" and meta.get("category",_category_for_rel(rel))!=category and _vectors is not None:
                    keep_vectors.append(_vectors[idx]);keep_meta.append(meta)
                    if rel in _manifest:new_manifest[rel]=_manifest[rel]
            for rel,path in scope.items():
                fp=_fingerprint(path); unchanged=not full and rel in _manifest and _manifest[rel].get("size")==fp["size"] and _manifest[rel].get("mtime_ns")==fp["mtime_ns"] and rel in old
                if unchanged and _vectors is not None:
                    idx,meta=old[rel];keep_vectors.append(_vectors[idx]);keep_meta.append(meta);new_manifest[rel]=_manifest[rel]
                else: changed.append((rel,path))
            total=len(changed);done=0
        _status={"message":f"正在{mode_text}（{category}）","done":done,"total":total,"mode":mode,"paused":False,"category":category};batch_no=0
        for offset in range(0,len(changed),max(1,_batch_size)):
            if _stop_event.is_set(): _save_checkpoint(keep_vectors,keep_meta,new_manifest,[r for r,_ in changed[offset:]],done,total,full,category);_status={"message":f"已停止，可從 {done}/{total} 繼續","done":done,"total":total,"mode":"stopped","paused":False,"category":category};return
            while _pause_event.is_set():
                _status={"message":f"{mode_text}已暫停：{done}/{total}","done":done,"total":total,"mode":mode,"paused":True,"category":category}
                if _stop_event.wait(.25): _save_checkpoint(keep_vectors,keep_meta,new_manifest,[r for r,_ in changed[offset:]],done,total,full,category);return
            chunk=changed[offset:offset+max(1,_batch_size)];vecs=_encode_paths([p for _,p in chunk])
            for (rel,path),vec in zip(chunk,vecs):
                if vec is None: continue
                model=_find_model_for_image(path);keep_vectors.append(vec);keep_meta.append({"image": rel,
                    "model": str(model.relative_to(root)).replace("\\", "/") if model else None,
                    "name": path.stem,
                    "folder": str(path.parent.relative_to(root)).replace("\\", "/"),
                    "category": _category_info_for_rel(rel)[0],
                    "subcategory": _category_info_for_rel(rel)[1],
                    "category_path": _category_info_for_rel(rel)[2]});new_manifest[rel]={**_fingerprint(path),"index_key":rel}
            done=min(total,done+len(chunk));batch_no+=1;pending=[r for r,_ in changed[offset+len(chunk):]]
            if batch_no%_checkpoint_every==0 or not pending:_save_checkpoint(keep_vectors,keep_meta,new_manifest,pending,done,total,full,category)
            _status={"message":f"正在{mode_text}（{category}）：{done}/{total}","done":done,"total":total,"mode":mode,"paused":False,"category":category}
        order=np.argsort([m["image"].lower() for m in keep_meta]).tolist() if keep_meta else [];keep_meta=[keep_meta[i] for i in order];dim=keep_vectors[0].shape[0] if keep_vectors else 512;matrix=np.vstack([keep_vectors[i] for i in order]).astype("float32") if order else np.empty((0,dim),dtype="float32");new_manifest={m["image"]:new_manifest[m["image"]] for m in keep_meta if m["image"] in new_manifest};_atomic_save(matrix,keep_meta,new_manifest)
        with _index_lock:_vectors,_metadata,_manifest=matrix,keep_meta,new_manifest
        _clear_checkpoint();_status={"message":f"{mode_text}完成（{category}），目前共 {len(keep_meta)} 張","done":total,"total":total,"mode":"idle","paused":False,"category":category}
    except Exception as exc:_status={"message":f"建立索引失敗：{exc}","done":_status.get("done",0),"total":_status.get("total",0),"mode":"error","paused":False,"category":category}
    finally:_indexing=False;_pause_event.clear();_stop_event.clear()


_load_index()

@app.get("/",response_class=HTMLResponse)
def home(): return FileResponse(APP_DIR/"static"/"index.html")

@app.get("/api/status")
def status():
    lib=active_library();root=library_root();cats=sorted({m.get("category",_category_for_rel(m.get("image",""))) for m in _metadata})
    if root.exists():
        cats=sorted(set(cats)|{p.name for p in root.iterdir() if p.is_dir()})
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in root.iterdir()):cats.append(UNCATEGORIZED)
    return {**_status,"indexing":_indexing,"library":str(root),"active_library_id":lib["id"],"active_library_name":lib["name"],"libraries":_config["libraries"],"device":_device,"gpu_name":_gpu_name,"batch_size":_batch_size,"indexed":len(_metadata),"categories":sorted(set(cats)),"resumable":_checkpoint_info()}

@app.post("/api/reindex")
def reindex(full:bool=False,category:str="全部"):
    global _indexing
    if _indexing:return {"message":"索引正在建立中"}
    if not library_root().exists():raise HTTPException(400,"模型庫不存在")
    _indexing=True;threading.Thread(target=_build_index_worker,kwargs={"full":full,"category":category},daemon=True).start();return {"message":"已開始"}

@app.post("/api/index/resume-checkpoint")
def resume_checkpoint():
    global _indexing
    if _indexing:return {"message":"索引正在建立中"}
    if not _checkpoint_info():raise HTTPException(400,"沒有可繼續的進度")
    _indexing=True;threading.Thread(target=_build_index_worker,kwargs={"resume":True},daemon=True).start();return {"message":"已繼續"}

@app.post("/api/index/discard-checkpoint")
def discard_checkpoint():
    if _indexing:raise HTTPException(400,"索引工作進行中")
    _clear_checkpoint();return {"message":"已捨棄"}
@app.post("/api/index/pause")
def pause_index():_pause_event.set();return {"message":"已要求暫停"}
@app.post("/api/index/resume")
def resume_index():_pause_event.clear();return {"message":"已繼續"}
@app.post("/api/index/stop")
def stop_index():_stop_event.set();_pause_event.clear();return {"message":"已要求停止並保存"}


def _choose_folder(title="請選擇 3D 模型庫") -> Path|None:
    script=f'''Add-Type -AssemblyName System.Windows.Forms;$d=New-Object System.Windows.Forms.FolderBrowserDialog;$d.Description='{title}';if($d.ShowDialog()-eq 'OK'){{[Console]::OutputEncoding=[Text.Encoding]::UTF8;Write-Output $d.SelectedPath}}'''
    r=subprocess.run(["powershell.exe","-NoProfile","-STA","-Command",script],capture_output=True,text=True,encoding="utf-8",errors="replace")
    return Path(r.stdout.strip().splitlines()[-1]).resolve() if r.returncode==0 and r.stdout.strip() else None

@app.post("/api/libraries/add")
def add_library(name:str=""):
    if _indexing:raise HTTPException(400,"請先停止索引工作")
    path=_choose_folder("選擇要新增的模型庫")
    if not path:return {"message":"已取消"}
    if any(Path(x["path"]).resolve()==path for x in _config["libraries"]):raise HTTPException(400,"這個路徑已經加入")
    lib={"id":_slug(name or path.name),"name":name.strip() or path.name,"path":str(path)};_config["libraries"].append(lib);_config["active_library_id"]=lib["id"];save_config();_load_index();return {"message":f"已新增並切換至：{lib['name']}"}

@app.post("/api/libraries/switch")
def switch_library(library_id:str):
    global _vectors,_metadata,_manifest
    if _indexing:raise HTTPException(400,"請先停止索引工作")
    _library_by_id(library_id);_config["active_library_id"]=library_id;save_config();_load_index();return {"message":f"已切換至：{active_library()['name']}"}

@app.post("/api/libraries/rename")
def rename_library(library_id:str,name:str):
    lib=_library_by_id(library_id);lib["name"]=name.strip() or lib["name"];save_config();return {"message":"已重新命名"}

@app.post("/api/libraries/remap")
def remap_library(library_id:str):
    if _indexing:raise HTTPException(400,"請先停止索引工作")
    lib=_library_by_id(library_id);path=_choose_folder("重新指定模型庫位置（保留索引）")
    if not path:return {"message":"已取消"}
    lib["path"]=str(path);save_config();return {"message":f"已重新對應：{path}"}

@app.delete("/api/libraries/{library_id}")
def delete_library(library_id:str,delete_index:bool=False):
    if _indexing:raise HTTPException(400,"請先停止索引工作")
    if len(_config["libraries"])<=1:raise HTTPException(400,"至少必須保留一個模型庫")
    _library_by_id(library_id);_config["libraries"]=[x for x in _config["libraries"] if x["id"]!=library_id]
    if delete_index:shutil.rmtree(LIBRARIES_DIR/library_id,ignore_errors=True)
    if _config["active_library_id"]==library_id:_config["active_library_id"]=_config["libraries"][0]["id"]
    save_config();_load_index();return {"message":"已刪除模型庫設定"}


def _load_library_index(lib_id:str):
    p=_paths_for(lib_id)
    if not (p["index"].exists() and p["meta"].exists()):return None,[]
    try:return np.load(p["index"])["vectors"].astype("float32"),json.loads(p["meta"].read_text(encoding="utf-8"))
    except Exception:return None,[]

@app.post("/api/search")
async def search(file:UploadFile=File(...),limit:int=24,category:str="全部",library_id:str=""):
    try: img=Image.open(file.file);qvec=_encode_image(img)
    except Exception as exc:raise HTTPException(400,f"無法讀取圖片：{exc}")
    targets=_config["libraries"] if library_id==ALL_LIBRARIES else [_library_by_id(library_id or None)]
    all_results=[]
    for lib in targets:
        vecs,meta=(_vectors,_metadata) if lib["id"]==active_library()["id"] else _load_library_index(lib["id"])
        if vecs is None or not len(meta):continue
        ids=[i for i,m in enumerate(meta) if category=="全部" or m.get("category",_category_for_rel(m.get("image","")))==category]
        if not ids:continue
        sims=vecs[ids]@qvec;top=np.argsort(-sims)[:limit]
        for pos in top:
            i=ids[int(pos)];m=meta[i];score=float(sims[int(pos)]);all_results.append((score,lib,i,m))
    all_results=sorted(all_results,key=lambda x:-x[0])[:limit]
    out=[]
    for score,lib,i,m in all_results:
        out.append({"result_id":i,"library_id":lib["id"],"library_name":lib["name"],"name":m.get("name"),"category":m.get("category",UNCATEGORIZED),"subcategory":m.get("subcategory",""),"category_path":m.get("category_path",m.get("category",UNCATEGORIZED)),"image":m.get("image"),"model":m.get("model"),"score":round(score*100,1),"preview_url":f"/api/preview/{lib['id']}/{i}","model_url":f"/api/model/{lib['id']}/{i}" if m.get("model") else None})
    return {"results":out}


def _item(lib_id:str,idx:int):
    lib=_library_by_id(lib_id);vecs,meta=(_vectors,_metadata) if lib_id==active_library()["id"] else _load_library_index(lib_id)
    if idx<0 or idx>=len(meta):raise HTTPException(404,"結果不存在")
    return lib,meta[idx]

@app.get("/api/preview/{lib_id}/{idx}")
def preview(lib_id:str,idx:int):
    lib,m=_item(lib_id,idx);p=Path(lib["path"])/Path(m["image"])
    if not p.exists():raise HTTPException(404,"圖片不存在")
    return FileResponse(p)
@app.get("/api/model/{lib_id}/{idx}")
def model_file(lib_id:str,idx:int):
    lib,m=_item(lib_id,idx)
    if not m.get("model"):raise HTTPException(404,"沒有模型檔")
    p=Path(lib["path"])/Path(m["model"])
    if not p.exists():raise HTTPException(404,"模型檔不存在")
    if p.is_dir():raise HTTPException(400,"對應項目是資料夾，請使用「開啟檔案位置」")
    return FileResponse(p,filename=p.name)
@app.post("/api/open-location/{lib_id}/{idx}")
def open_location(lib_id:str,idx:int):
    lib,m=_item(lib_id,idx);root=Path(lib["path"]);p=root/Path(m.get("model") or m["image"])
    if not p.exists():raise HTTPException(404,"檔案不存在")
    if os.name!="nt":raise HTTPException(400,"此功能僅支援 Windows")
    subprocess.Popen(["explorer.exe","/select,",str(p)]);return {"message":"已開啟"}
