"""量「取一次眼睛画面」要多少毫秒 —— 顺手看渲染到底有没有落在显卡上。

为什么专门有这个探针：云镜像常常只装了驱动里跑 CUDA 的那半，没装渲染的那半
（libEGL_nvidia）。这时 MUJOCO_GL=egl 不会报错，会安安静静落到 Mesa 的软件光栅上，
nvidia-smi 里 GPU 几乎是闲的，而每取一次眼睛画面要几百毫秒。同一台 4090 上量到过：

    软件光栅   464.63 ms/帧    一只孩子跑 10 秒要 55 秒
    上了卡       2.57 ms/帧    一只孩子约 10 秒

差 177 倍，而且不报错 —— 所以这个数字要定期亲自看一眼，别信「跑得起来就是好的」。
本机（Windows，wgl）同一段代码是 2.6 ms 左右，可以作为尺子。

渲染库怎么补：见 cloud/gpu_render.sh。

    python tools/probe_render_speed.py [帧数] [--width 192 --height 144]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.go2_body import Go2Body
from born_wired.stereo_senses import RawEyes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", nargs="?", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--model", type=Path, default=ROOT / "models/reflex_arena.xml")
    args = parser.parse_args()
    body = Go2Body(args.model)
    eyes = RawEyes(body, width=args.width, height=args.height)
    for _ in range(args.warmup):
        eyes.observe_raw()
    started = time.perf_counter()
    for _ in range(args.frames):
        eyes.observe_raw()
    per_frame = (time.perf_counter() - started) / args.frames * 1000
    eyes.close()
    print("眼 %d x %d" % (args.width, args.height))
    print("取一次眼睛画面  %.2f ms" % per_frame)
    if per_frame > 50:
        print("这个数太大：渲染多半掉到软件光栅上了（跑起来不报错，但慢一两百倍）。")
        print("显卡机上先跑 bash cloud/gpu_render.sh，再量一遍。")
    else:
        print("正常区间（显卡上 1~5 ms，本机 wgl 约 2.6 ms）。")


if __name__ == "__main__":
    main()
