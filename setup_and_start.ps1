$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Python = Join-Path $Runtime 'python.exe'
$LogDir = Join-Path $Root 'logs'
$SetupLog = Join-Path $LogDir 'setup.log'
$PyVersion = '3.11.9'
$Zip = Join-Path $env:TEMP ("python-{0}-embed-amd64.zip" -f $PyVersion)
$Url = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
try { Start-Transcript -Path $SetupLog -Append | Out-Null } catch {}
function Stop-WithError([string]$Message) {
  Write-Host ''; Write-Host ('ERROR: ' + $Message) -ForegroundColor Red; Write-Host ('Log: ' + $SetupLog)
  try { Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show(($Message + "`r`n`r`nLog: " + $SetupLog),'3D Model Image Search',[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Error) | Out-Null } catch {}
  try { Stop-Transcript | Out-Null } catch {}; exit 1
}
try {
  Write-Host ('Root: ' + $Root)
  if (-not (Test-Path $Python)) {
    Write-Host 'Step 1/4: Downloading portable Python...'
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
    Expand-Archive -Path $Zip -DestinationPath $Runtime -Force
    $pth = Get-ChildItem $Runtime -Filter 'python*._pth' | Select-Object -First 1
    if ($null -eq $pth) { throw 'Python _pth file was not found.' }
    $content = Get-Content $pth.FullName; $content = $content -replace '^#import site$', 'import site'; if ($content -notcontains '..') { $content += '..' }
    Set-Content -Path $pth.FullName -Value $content -Encoding ASCII
    $GetPip = Join-Path $Runtime 'get-pip.py'; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $GetPip -UseBasicParsing
    & $Python $GetPip; if ($LASTEXITCODE -ne 0) { throw ('pip installation failed: ' + $LASTEXITCODE) }
  } else { Write-Host 'Step 1/4: Portable Python already exists.' }

  Write-Host 'Step 2/4: Detecting NVIDIA GPU...'
  $HasNvidia = $false
  try { $null = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null; if ($LASTEXITCODE -eq 0) { $HasNvidia = $true } } catch {}
  $Wanted = if ($HasNvidia) { 'cuda' } else { 'cpu' }
  $Marker = Join-Path $Runtime ('.dependencies_ready_' + $Wanted)
  if (-not (Test-Path $Marker)) {
    Write-Host ('Step 3/4: Installing packages for ' + $Wanted + '. This can take a long time...')
    & $Python -m pip install --upgrade pip
    & $Python -m pip install --no-warn-script-location -r (Join-Path $Root 'requirements-portable.txt')
    if ($LASTEXITCODE -ne 0) { throw ('Package installation failed: ' + $LASTEXITCODE) }
    if ($HasNvidia) {
      & $Python -m pip install --upgrade --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    } else {
      & $Python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
    }
    if ($LASTEXITCODE -ne 0) { throw ('PyTorch installation failed: ' + $LASTEXITCODE) }
    Get-ChildItem $Runtime -Filter '.dependencies_ready_*' -ErrorAction SilentlyContinue | Remove-Item -Force
    New-Item -ItemType File -Force -Path $Marker | Out-Null
  } else { Write-Host ('Step 3/4: ' + $Wanted + ' packages already exist.') }

  $CudaCheck = & $Python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
  Write-Host ('PyTorch device: ' + $CudaCheck)
  Write-Host 'Step 4/4: Starting application...'
  & $Python (Join-Path $Root 'app\portable_launcher.py')
  $Code = $LASTEXITCODE; if ($Code -ne 0) { throw ('Launcher stopped with code: ' + $Code) }
} catch { Stop-WithError $_.Exception.Message }
try { Stop-Transcript | Out-Null } catch {}; exit 0
