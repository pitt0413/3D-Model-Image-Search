from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.json"
LOG_DIR = ROOT / "logs"
APP_LOG = LOG_DIR / "app.log"
LAUNCH_LOG = LOG_DIR / "launcher.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LAUNCH_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def msgbox(text: str, title: str = "3D 模型以圖搜圖", error: bool = False) -> None:
    flags = 0x10 if error else 0x40
    ctypes.windll.user32.MessageBoxW(None, text, title, flags)


def load_config() -> dict:
    if CONFIG.exists():
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            log(f"讀取 config.json 失敗：{exc}")
    return {"library_dir": "", "port": 8090, "search_limit": 24}


def save_config(cfg: dict) -> None:
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_folder() -> str:
    script = r'''
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '請選擇 3D 模型庫資料夾'
$dialog.ShowNewFolderButton = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.SelectedPath
}
'''
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        log(f"資料夾選擇器錯誤：{result.stderr}")
        return ""
    return result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def is_our_app(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1.5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return isinstance(payload, dict) and ("library" in payload or "library_dir" in payload) and "device" in payload
    except Exception:
        return False


def find_available_port(preferred: int) -> int:
    for port in range(preferred, preferred + 101):
        if not port_open(port):
            return port
    raise RuntimeError(f"找不到可用連接埠（已檢查 {preferred}～{preferred + 100}）")


def main() -> int:
    log("啟動器開始執行")
    cfg = load_config()
    library = str(cfg.get("library_dir", "")).strip()

    if not library or not Path(library).is_dir():
        msgbox("第一次使用，請選擇你的 3D 模型庫資料夾。\n\n資料夾內可包含渲染圖片與 SKP、MAX、FBX 等模型檔。")
        library = choose_folder()
        if not library:
            msgbox("尚未選擇模型庫，程式未啟動。", error=True)
            log("使用者未選擇模型庫")
            return 2
        cfg["library_dir"] = library
        save_config(cfg)
        log(f"模型庫設定為：{library}")

    preferred_port = int(cfg.get("port", 8090))
    port = preferred_port

    if port_open(port):
        if is_our_app(port):
            url = f"http://127.0.0.1:{port}"
            log(f"偵測到本 App 已在連接埠 {port} 執行，直接開啟瀏覽器")
            webbrowser.open(url)
            return 0
        port = find_available_port(preferred_port + 1)
        cfg["port"] = port
        save_config(cfg)
        log(f"連接埠 {preferred_port} 被其他程式占用，自動改用 {port}")

    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["LIBRARY_DIR"] = library
    env["DATA_DIR"] = str(ROOT / "data")
    env["DEVICE_MODE"] = str(cfg.get("device", "auto"))

    app_log = APP_LOG.open("a", encoding="utf-8")
    command = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port)
    ]
    log(f"啟動服務：{' '.join(command)}")
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=app_log,
        stderr=app_log,
    )

    started = False
    for _ in range(600):
        if proc.poll() is not None:
            msgbox(f"程式啟動失敗。\n\n請查看：\n{APP_LOG}", error=True)
            log(f"服務提前結束，代碼：{proc.returncode}")
            app_log.close()
            return 3
        if is_our_app(port):
            started = True
            break
        time.sleep(1)

    if not started:
        proc.terminate()
        msgbox(f"啟動逾時。\n\n請查看：\n{APP_LOG}", error=True)
        log("服務啟動逾時")
        app_log.close()
        return 4

    webbrowser.open(url)
    print("\n" + "=" * 62)
    print("  3D 模型以圖搜圖已啟動")
    print(f"  模型庫：{library}")
    print(f"  搜尋頁面：{url}")
    print("\n  請保持這個視窗開啟。")
    print("  要關閉程式時，回到此視窗按 Enter。")
    print("=" * 62 + "\n")

    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        log("收到關閉指令")
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        app_log.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"未處理錯誤：{type(exc).__name__}: {exc}")
        msgbox(f"啟動失敗：{exc}\n\n請查看：\n{LAUNCH_LOG}", error=True)
        raise
