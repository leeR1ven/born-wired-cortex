# -*- coding: utf-8 -*-
# 把云端某一批训练的结果拉回本地做备份。
#   用法：  powershell -NoProfile -ExecutionPolicy Bypass -File _pull_cloud.ps1 七批
# 不传参数就拉「七批」。
param([string]$batch = "七批")
$ErrorActionPreference = "SilentlyContinue"
$key  = "C:\Users\Administrator\.ssh\id_gpushare2"
$dest = "F:\born-wired-cortex\engine_v2\artifacts\云端\$batch"
$log  = "F:\born-wired-cortex\engine_v2\artifacts\_pull_cloud.log"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$opts = @("-i", $key, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "-P", "43328")
$out  = & scp @opts "root@i-2.gpushare.com:/hy-tmp/bwc/engine_v2/artifacts/追球_云_${batch}_*" $dest 2>&1
$out += & scp @opts "root@i-2.gpushare.com:/hy-tmp/bwc/logs/追球_云_${batch}_*.log" $dest 2>&1
$out += & scp @opts "root@i-2.gpushare.com:/hy-tmp/bwc/logs/watchdog.log" "$dest\watchdog.log" 2>&1
$text = ($out | Out-String).Trim()
if ($text.Length -gt 400) { $text = $text.Substring(0, 400) }
Add-Content -Path $log -Value ("[$stamp] $batch :: " + ($text -replace "`r?`n", " | "))
Write-Host "拉到 $dest"