$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root

Write-Host "[1/4] Preparing installer assets..." -ForegroundColor Cyan
uv run --no-sync python installer\prepare_assets.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/4] Building PyInstaller bundle..." -ForegroundColor Cyan
uv run --no-sync --with pyinstaller pyinstaller --noconfirm --clean installer\OcrPipeline.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/4] Ensuring WiX UI extension..." -ForegroundColor Cyan
$wixVersion = (wix --version).Split("+")[0].Trim()
$extList = wix extension list -g 2>$null
if (-not ($extList -match "WixToolset\.UI\.wixext")) {
    wix extension add -g "WixToolset.UI.wixext/$wixVersion"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[4/4] Building Windows MSI Installation Wizard..." -ForegroundColor Cyan
$msiOut = Join-Path $Root "dist\OcrPipeline-0.1.0-x64.msi"
wix build `
    -arch x64 `
    -ext WixToolset.UI.wixext `
    -o $msiOut `
    installer\windows\OcrPipeline.wxs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done! Installer created at: $msiOut" -ForegroundColor Green
