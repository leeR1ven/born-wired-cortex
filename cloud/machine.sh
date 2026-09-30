#!/usr/bin/env bash
# 这台机器「真正」能用多少核、多少内存。
#
# bench.sh 和 run.sh 都从这里取数，别再直接用 nproc / free：容器里 nproc 报的是宿主机的
# 核数（恒源云那台 4090 报 96 核），free 报的是宿主机的内存（377 GB），而 cgroup 给的配额
# 才是我们能用的（那台实际是 24 核 / 124 GB）。按宿主机的数字开进程，CPU 会互相抢，
# 内存也会被 OOM 干掉 —— 96 个进程 × 每只约 1.9 GB = 182 GB，直接超。

cpu_cores() {
    local quota period
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max || true
        if [ -n "${quota:-}" ] && [ "$quota" != "max" ]; then
            echo $(( quota / period ))
            return
        fi
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "${quota:-0}" -gt 0 ]; then
            echo $(( quota / period ))
            return
        fi
    fi
    nproc
}

mem_gb() {
    local limit=0
    if [ -r /sys/fs/cgroup/memory.max ]; then
        limit=$(cat /sys/fs/cgroup/memory.max)
    elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    fi
    # cgroup v2 的「不限」是个很大的数，别把它当成 8 EB 的内存
    [ "${limit:-0}" -gt 1099511627776 ] 2>/dev/null && limit=0
    if [ "$limit" -gt 0 ] 2>/dev/null; then
        echo $(( limit / 1024 / 1024 / 1024 ))
    else
        free -g | awk "/^Mem:/{print \$2}"
    fi
}
