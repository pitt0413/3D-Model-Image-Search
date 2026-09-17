# Privacy Check

公開 GitHub 前已針對 v1.6.0 專案檢查常見敏感資訊。

## 未發現

- API Key
- Password
- Access Token / Bearer Token
- Email
- 私人 IP
- 寫死的 Windows 使用者目錄
- 個人 NAS 位址

## 執行後可能產生的私人資訊

### `config.json`

可能保存使用者所選的模型庫絕對路徑，例如 Windows 磁碟或 NAS 路徑。

### `logs/`

Log 與 PowerShell transcript 可能保存：

- Windows 使用者或環境資訊
- App 所在路徑
- 模型庫所在路徑
- 執行錯誤資訊

### `data/`

索引資料可能透露：

- 私人模型檔名
- 模型分類名稱
- 模型庫資料夾結構

### `runtime/`

主要為下載後的 Python、PyTorch 與依賴套件，本身不是私人資料，但體積很大且不適合提交 Repository。

## 保護措施

`.gitignore` 已排除上述本機資料。公開 Repository 時請勿使用 `git add -f` 強制提交它們。
