ls /hy-tmp/bwc/cloud/ 2>/dev/null | head
grep -rn "python" /hy-tmp/bwc/cloud/*.sh 2>/dev/null | head -20
ls /opt/conda/bin/python* 2>/dev/null
which -a python python3 2>/dev/null
/opt/conda/bin/python -c "import numpy, mujoco; print('conda ok')" 2>&1 | tail -2