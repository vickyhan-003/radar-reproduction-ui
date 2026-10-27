$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\1\miniconda3\envs\radar\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到 radar 环境：$Python"
}

Set-Location -LiteralPath $ProjectRoot
& $Python -m streamlit run "$ProjectRoot\radar_ui.py"
