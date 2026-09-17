# 3D Model Image Search

**3D 模型以圖搜圖工具 · v1.6.0**

Windows 免安裝工具。使用 OpenCLIP 為 3D 模型預覽圖建立向量索引，之後可直接用圖片搜尋相似的 3D 模型。

> 第一次啟動需要網路連線，程式會自動建立 Portable Python 執行環境並安裝必要套件。完成後，日常使用不需要另外安裝 Python。

## 主要功能

- 使用圖片搜尋相似 3D 模型。
- 支援任意層級的模型分類資料夾。
- 自動辨識模型預覽圖。
- 智慧排除模型解壓縮資料夾內的材質貼圖。
- 自動排除 `textures`、`maps`、`materials`、`貼圖`、`材質` 等常見貼圖資料夾。
- 自動排除 `diffuse`、`normal`、`roughness`、`bump`、`albedo` 等常見材質貼圖名稱。
- 搜尋結果顯示完整分類路徑。
- NVIDIA 顯示卡環境可自動安裝 CUDA 版 PyTorch；其他環境使用 CPU 版。
- 不需要傳送模型庫到雲端，索引與搜尋都在本機執行。

## 系統需求

- Windows 10 / 11 64-bit
- 可使用 PowerShell
- 第一次啟動需要網路連線
- 建議預留數 GB 磁碟空間給 Portable Python、PyTorch 與模型權重
- NVIDIA GPU 可加速；沒有 NVIDIA GPU 也可使用 CPU

## 快速開始

1. 下載專案 ZIP，或 Clone Repository。
2. 解壓縮到你要保存的位置。
3. 雙擊 `START.cmd`。
4. 第一次啟動會自動：
   - 下載 Portable Python 3.11.9
   - 安裝必要 Python 套件
   - 偵測 NVIDIA GPU
   - 安裝對應的 PyTorch
5. 程式啟動後選擇你的 3D 模型庫資料夾。
6. 建立索引後即可開始以圖搜圖。

第一次完成環境準備後，之後仍然只需要雙擊 `START.cmd`。

## 模型庫範例

```text
傢俱
├─ 沙發
│  ├─ 沙發001.jpg
│  ├─ 沙發001.zip
│  └─ 沙發001
│     ├─ 沙發001.skp
│     └─ textures
│        ├─ fabric_diffuse.jpg
│        └─ fabric_normal.jpg
├─ 床
└─ 櫃
```

會建立索引：

```text
傢俱\沙發\沙發001.jpg
```

不會建立索引：

```text
傢俱\沙發\沙發001\textures\fabric_diffuse.jpg
傢俱\沙發\沙發001\textures\fabric_normal.jpg
```

## 專案結構

```text
3D-model-image-search
├─ app/
│  ├─ main.py
│  ├─ portable_launcher.py
│  └─ static/
│     └─ index.html
├─ data/                  # 本機索引與快取，不提交 Git
├─ runtime/               # 第一次啟動後下載的 Python / 套件，不提交 Git
├─ START.cmd
├─ setup_and_start.ps1
├─ requirements-portable.txt
├─ config.example.json
└─ .gitignore
```

## 隱私與 GitHub 注意事項

程式執行後會在本機產生一些不適合公開提交的資料：

- `config.json`：可能包含模型庫的本機或 NAS 絕對路徑。
- `logs/`：可能包含程式路徑、Windows 環境與模型庫路徑。
- `data/`：索引可能透露模型檔名、分類名稱與資料夾結構。
- `runtime/`：Portable Python、PyTorch 與相關套件，體積很大。

本專案的 `.gitignore` 已經預設排除以上內容。公開 Repository 前不要手動強制加入這些檔案。

詳細檢查結果請參考 [`PRIVACY_CHECK.md`](PRIVACY_CHECK.md)。

## 從 v1.5.0 升級

如果已有舊版，可複製以下資料到新版：

- `runtime/`
- `data/`
- `config.json`

由於 v1.6.0 新增智慧排除材質貼圖的規則，舊索引可能已包含材質圖，因此升級後建議對每套模型庫執行一次「完整重建索引」。

## 疑難排解

### 第一次啟動很久

正常。第一次需要下載 Python、PyTorch、OpenCLIP 相關套件與模型權重，之後就不需要重複安裝。

### 啟動失敗

查看 `logs/setup.log`。如果要回報問題，建議先把其中的 Windows 使用者名稱、本機路徑、NAS 路徑等私人資訊遮掉。

### 換電腦使用

直接複製專案即可。如果不複製 `runtime/`，新電腦第一次執行 `START.cmd` 時會重新建立環境。

## 版本

目前版本：**v1.6.0**

完整變更請參考 [`CHANGELOG.md`](CHANGELOG.md)。

## License

本專案採用 [MIT License](LICENSE)。

Copyright © 2026 Pitt Tools

## 開發說明

本工具由 Pitt Tools 製作，開發過程使用 ChatGPT 協助程式設計、除錯與文件整理。
