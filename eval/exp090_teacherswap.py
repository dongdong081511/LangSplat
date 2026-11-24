# exp090_teacherswap.py — E5 teacher-swap (P3 备选): 优势来自教师身份还是场管线?
#
# 用 CLIP image encoder 作教师、同 AE 架构同维度 (32d)、同 3DGS 场管线训出的既有场
# (teatime/ramen/waldo_kitchen/figurines_clip32d, EXP-017/018 维度曲线 32d 点),
# 以 exp082 穷举帧对协议逐位复用评测: 2D 教师 (原生 CLIP 512d) vs 3D CLIP 场.
# 对照 = exp082 的 DINO 行 (2D DINO 768d → 3D DINO 32d 场).
#   CLIP lift ≈ DINO lift → 优势主体 = 场管线 (多视角聚合);
#   CLIP lift ≪ DINO lift → 教师特征属性 (DINOv2 判别空间) 是必要条件.
# 前置: run_exp090_e5_teacherswap_prep.sh (重提原生 CLIP 特征 + 重渲 clip32d 场).
# 用法: python eval/exp090_teacherswap.py 2>&1 | tee eval_result/exp090_teacherswap.log  (纯 CPU)
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp082_attribution as E

E.SCENES = [
    dict(scene='teatime',       feat='clip', field='teatime_clip32d_1'),
    dict(scene='figurines',     feat='clip', field='figurines_clip32d_1'),
    dict(scene='ramen',         feat='clip', field='ramen_clip32d_1'),
    dict(scene='waldo_kitchen', feat='clip', field='waldo_kitchen_clip32d_1'),
]

if __name__ == '__main__':
    for cfg in E.SCENES:
        try:
            E.run_scene(cfg)
        except Exception as e:
            print(f"[{cfg['scene']}] ERROR: {e}")
            traceback.print_exc()
