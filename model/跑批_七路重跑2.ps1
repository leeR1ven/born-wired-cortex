
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

# ---- R1 的白脑对照（本能表清空），这一步要在"前额叶不保留"下跑 ----
$env:前额叶不保留 = "1"
跑 "实验_白脑对照.py"            "日志_白脑_新.log"              ""
Remove-Item Env:前额叶不保留

# ---- R2 的第二块：带环（不设 前额叶不保留）----
跑 "实验_两半各亮多少.py"         "日志_两半_新.log"              ""

# ---- R7 的破坏版 ----
跑 "实验_模糊化_存活曲线.py"      "日志_存活曲线_新.log"          ""

# ---- R5：只跑视觉层，不建整片皮层（几个种子，很快） ----
跑 "实验_方位分辨力_按层.py"      "日志_方位分辨力.log"           ""
跑 "实验_前几层看得见小东西吗.py" "日志_前几层.log"               ""
跑 "实验_追踪靠的是哪一层.py"     "日志_分层.log"                 ""

# ---- 5.4 / 5.5 / 效率那几个小诊断 ----
跑 "诊断_步态相位.py"            "日志_步态相位.log"             ""
跑 "诊断_整拍耗时.py"            "日志_整拍耗时.log"             ""
跑 "验证_学抑制_逐拍对照.py"      "日志_验证学抑制_逐拍.log"       ""
跑 "对照_学抑制_改前改后.py"      "日志_对照_学抑制_改前改后.log"  ""
跑 "对照_现成RL_驱动.py"          "日志_对照_现成RL.log"          ""

Write-Host "第二批跑完"
