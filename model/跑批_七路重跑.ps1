
$ErrorActionPreference = "Continue"
$py = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
Set-Location "F:\born-wired-cortex\model"
$env:PYTHONIOENCODING = "utf-8"
$env:前额叶不保留 = "1"

function 跑($脚本, $日志, $参数) {
    Write-Host ("=== " + $脚本 + " === " + $日志)
    $a = @("-X", "utf8", $脚本)
    if ($参数) { $a += $参数 }
    $p = Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -Wait -PassThru `
         -RedirectStandardOutput $日志 -RedirectStandardError ($日志 + ".err.txt")
    Write-Host ("    退出码 " + $p.ExitCode)
}

跑 "实验_闭环前提_多种子.py"        "日志_闭环多种子_新.log"         ""
跑 "实验_前额叶_多种子.py"          "日志_前额叶多种子_新A.log"      ""
跑 "实验_两半各亮多少.py"           "日志_两半_无环.log"             ""
跑 "实验_本能分阶段.py"             "日志_分阶段_新.log"             ""
跑 "实验_想还在吗.py"               "日志_想还在吗_无环.log"         ""
跑 "实验_切断回响.py"               "日志_切断回响_有环.log"         ""

Remove-Item Env:前额叶不保留

跑 "实验_眼睛跟随.py"               "日志_眼睛跟随.log"              "小横着飘"
跑 "实验_本能表整体偏移_正前方.py"  "日志_整体偏移_正前方_新.log"    ""
跑 "诊断_天生步态稳不稳.py"         "日志_天生步态.log"              ""

Write-Host "全部跑完"
