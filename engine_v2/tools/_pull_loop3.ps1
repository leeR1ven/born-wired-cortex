# 每隔 15 分钟把云端三批的结果拉回本地做备份，跑满 6 小时后自己退出
$ErrorActionPreference = "SilentlyContinue"
$key  = "C:\Users\Administrator\.ssh\id_gpushare2"
$dest = "F:\born-wired-cortex\engine_v2\artifacts\云端\三批"
$log  = "F:\born-wired-cortex\engine_v2\artifacts\_pull_cloud3.log"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$opts = @("-i", $key, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "-P", "43328")
$end = (Get-Date).AddHours(6)
while ((Get-Date) -lt $end) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $out = & scp @opts "root@i-2.gpushare.com:/hy-tmp/bwc/engine_v2/artifacts/追球_云_三批_*" $dest 2>&1
    $out += & scp @opts "root@i-2.gpushare.com:/hy-tmp/bwc/logs/追球_云_三批_*.log" $dest 2>&1
    $out += & scp @opts "root@i-2.gpushare.com:/hy-tmp/bwc/logs/watchdog.log" $dest 2>&1
    $text = ($out | Out-String).Trim()
    if ($text.Length -gt 300) { $text = $text.Substring(0, 300) }
    Add-Content -Path $log -Value ("[$stamp] " + ($text -replace "`r?`n", " | "))
    Start-Sleep -Seconds 900
}
Add-Content -Path $log -Value ("[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] 备份循环结束")