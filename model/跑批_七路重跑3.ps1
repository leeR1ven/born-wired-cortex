
$ErrorActionPreference = "Continue"
$py = "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
Set-Location "F:\born-wired-cortex\model"
$env:PYTHONIOENCODING = "utf-8"

function 跑($脚本, $日志, $参数) {
    Write-Host ("=== " + $脚本 + " === " + $日志)
    $a = @("-X", "utf8", $脚本)
    if ($参数) { $a += $参数.Split(" ") }
    $p = Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -Wait -PassThru `
         -RedirectStandardOutput $日志 -RedirectStandardError ($日志 + ".err.txt")
    Write-Host ("    退出码 " + $p.ExitCode)
}

# ---- R9：声音叫走路，四个阶段 ----
跑 "实验_发育_声音叫走路_多种子.py" "日志_发育多种子_A.log" ""
跑 "实验_发育_无奖励对照.py"       "日志_无奖励对照.log"   ""
跑 "实验_发育_无环对照.py"         "日志_无环对照.log"     ""

# ---- R9 的定位诊断 ----
跑 "诊断_摔是谁干的2.py"           "日志_摔是谁干的2.log"    "20260915 20260918"
跑 "诊断_摔是谁干的2.py"           "日志_摔是谁干的3_无环.log" "--无环 20260915 20260918"
跑 "诊断_前额叶打走路链.py"         "日志_前额叶打走路链.log"  "20260915"
跑 "诊断_冻死进运动.py"            "日志_冻死进运动.log"    "20260915"
跑 "诊断_返回线截断.py"            "日志_返回线截断.log"    "20260915"
跑 "试_打折R9_参数.py"             "日志_打折R9.log"        "0.75"

# ---- R4：色相边界 ----
跑 "诊断_色相_红块.py"             "日志_色相红块_新.log"   ""
跑 "实验_渐变_正式协议.py"          "日志_渐变正式_新.log"   "3,6,12 20260914,20260915,20260916,20260917,20260918"

# ---- R10：仿生奖励（触觉区）----
跑 "实验_发育_仿生奖励_三臂.py"      "日志_仿生奖励_一直摸.log"      "一直摸 20260914 20260915 20260916"
跑 "实验_发育_仿生奖励_声音叫走路.py" "日志_仿生奖励_声音叫走路.log"  "20260914,20260915,20260916"
跑 "诊断_触觉版_天生站起来.py"       "日志_触觉版_天生站起来.log"    "20260914 20260915 20260916"

Write-Host "第三批跑完"
