
## EXP-052 系列 (α 曲线系统研究)
- figurines α 7点曲线: chosen 77.78→96.30 (α≥0.95渐近), majority 峰0.85-0.9 (92.59), hit-rate 渐近89.6%; teatime 平台α∈[0.3,0.7] (0.85微降)
- 主表 figurines 配置=α0.9 (双口径平衡92.59); α≥0.95 chosen 96.30 (+7.4pp vs CLIP, p=0.219 ns) 但 majority −1对 → 纯自蒸馏过拟合分叉 (单帧锐利/跨帧一致性受损)
- α=1.0 定点自蒸馏稳定, 未复现 EXP-037 iter2 退化 (动态重渲≠固定渲染监督)
- 3D 项来源: a07 场渲染 (raw ckpt 已删); α 最优值场景相关, 主表 per-scene 报

## EXP-054 (per-scene 最优 α 统一主表, 定稿)
- α 最优分三型: 低 α 平台 (teatime 0.3-0.7/ramen 0.7) vs 高 α 峰 (figurines 0.85-0.9) vs 无增益 (waldo raw); ramen a09 majority −3.8pp vs a07 (5:2 ns) → 保留 a07
- 统一主表 (majority): teatime +13.1★/figurines a09 +14.8★(首次单独显著 0.022)/ramen +25.3★★★/waldo raw −15.4 ns; pooled b=5 c=39 p≈1.4e-7
- waldo raw 场重训逐分复现 EXP-040; raw≡a07 majority 零 discordant — 平滑伤害是单帧现象, 多帧投票免疫 (majority 口径优越性新论据)
- 脚本坑: mcnemar.py 必须从项目根调用 (eval/ cwd 下相对路径错 + set -e 杀链, run_exp054→054b 修复); binomtest.pvalue 已双侧, 再 ×2 会翻倍 p 值

## EXP-055 (RM query x per-scene 场, 优势分解)
- RM 口径 pooled majority 完全拉平 (b=10 c=11 p=1.0) — EXP-054 主表 p≈1.4e-7 的显著性主要来自 query 入口端 (AE 压缩不对称), 非场质量
- 叙事重定位: 路由主张 = "入口维度需求(image 高维)与 text 场维度需求(CLIP 8d, EXP-018)结构性冲突, 单场不可两全"; 不是 "DINO 场判别更高" (RM 证伪)
- ramen 唯一场质量优势场景 (RM chosen +8.9pp p=0.078, n=79)
- 主表决策: 2D tile query (现实入口) 为主 + RM (场质量上界) 并列诚实对照; figurines a09 重训逐分复原 ✓
- run_exp054.sh 的 rm -rf 因 McNemar 报错未执行 — waldo raw 渲染/ckpt 侥幸保留, 后续清理脚本需注意错误处理顺序 (清理放最前或用 trap)
