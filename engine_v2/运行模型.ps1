param(
    [double]$Seconds = 300,
    [switch]$Headless,
    [switch]$Demo
)
$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Python 3.13 was not found. See README.md to configure the interpreter.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Demo) { throw '旧版演示已归档到 releases/v0_2；当前模型没有动作播放脚本。' }
    $taskArguments = @('tools\live_dog.py', '--status', 'artifacts\live_v3_status.json')
    if ($Headless) {
        $taskArguments = @('tools\validate_reflex_v3.py', '--duration', [string]$Seconds,
                           '--seeds', '0', '--model', 'models\reflex_arena.xml',
                           '--output', 'artifacts\manual_headless_run.json')
    }
    & $taskPython @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Model exited with status $LASTEXITCODE" }
}
finally { Pop-Location }
