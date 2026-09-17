# Changelog

## v1.6.0 — 2026-09-17

### Added
- 支援任意層級的模型分類資料夾。
- 智慧辨識模型預覽圖。
- 自動排除與壓縮模型同名的解壓縮資料夾內圖片。
- 自動排除常見材質與貼圖資料夾。
- 自動排除 diffuse、normal、roughness、bump、albedo 等材質圖命名。
- 搜尋結果顯示完整分類路徑。

### Packaging
- 使用 `START.cmd` 作為 Windows Portable 啟動入口。
- 首次啟動自動下載 Portable Python 3.11.9。
- 自動偵測 NVIDIA GPU 並安裝 CUDA / CPU 對應 PyTorch。
- 加入 GitHub 用 `.gitignore`，排除私人設定、索引、Log 與 Runtime。

### Privacy
- 公開版本不包含使用者模型庫路徑、API Key、密碼或 Token。
