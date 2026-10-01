# -*- coding: utf-8 -*-
# 把这一批要用的三个工具 + 起点池推到云端，并核对 md5。
$key  = "C:\Users\Administrator\.ssh\id_gpushare2"
$host_ = "root@i-2.gpushare.com"
$port = "43328"
$src  = "F:\born-wired-cortex\engine_v2"
$dst  = "/hy-tmp/bwc/engine_v2"

$files = @(
  @{ local = "$src\tools\chase_red_ball.py"; remote = "$dst/tools/" },
  @{ local = "$src\tools\chase_task.py";     remote = "$dst/tools/" },
  @{ local = "$src\tools\evolve_chase.py";   remote = "$dst/tools/" },
  @{ local = "$src\artifacts\追球_云_三批_起点.jsonl"; remote = "$dst/artifacts/" }
)
foreach ($f in $files) {
  & scp -i $key -P $port -o StrictHostKeyChecking=no $f.local "${host_}:$($f.remote)" 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Host "上传失败：$($f.local)"; exit 1 }
  Write-Host "传好 $($f.local)"
}
Write-Host "`n=== 本地 md5 ==="
foreach ($f in $files) { (Get-FileHash $f.local -Algorithm MD5).Hash + "  " + (Split-Path $f.local -Leaf) }
Write-Host "`n=== 云端 md5 ==="
& ssh -i $key -p $port -o StrictHostKeyChecking=no $host_ "cd /hy-tmp/bwc/engine_v2 && md5sum tools/chase_red_ball.py tools/chase_task.py tools/evolve_chase.py artifacts/追球_云_三批_起点.jsonl && wc -l artifacts/追球_云_三批_起点.jsonl"