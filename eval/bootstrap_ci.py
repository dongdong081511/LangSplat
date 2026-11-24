# EXP-108 (P1-3) part A: bootstrap 95% CIs on the per-prompt text records
# collected in EXP-106 (no evaluator rerun needed; records are the exact
# per-frame per-prompt chosen-level IoUs and localization hits).
import json, os
import numpy as np

LSP = "/home/xiedexia/project/LangSplat/eval_result/pareto_locerr"
LAGA = "/home/xiedexia/project/LaGa_ref/eval_bridge"
SYSTEMS = {
    "ours": [("teatime", "records_teatime_8d.json"), ("figurines", "records_figurines_24d.json"),
             ("ramen", "records_ramen_clip8d.json"), ("waldo_kitchen", "records_waldo_kitchen_clip8d.json")],
    "3d": [("teatime", "records_teatime_3d.json"), ("figurines", "records_figurines_3d.json"),
           ("ramen", "records_ramen_3d.json"), ("waldo_kitchen", "records_waldo_kitchen_3d.json")],
    "laga": [("teatime", "records_text_teatime_noens.json"), ("figurines", "records_text_figurines_noens.json"),
             ("ramen", "records_text_ramen_noens.json"), ("waldo_kitchen", "records_text_waldo_kitchen_noens.json")],
}
rng = np.random.default_rng(42)
B = 10000

print(f"bootstrap 95% CI ({B} resamples, unit = frame x prompt record)")
print(f"{'system':>6} {'scene':<15} {'mIoU':>7} {'CI95':>18} {'mAcc':>7} {'CI95':>18}")
out = {}
for sys_name, files in SYSTEMS.items():
    for scene, fn in files:
        p = os.path.join(LSP if sys_name != "laga" else LAGA, fn)
        recs = json.load(open(p))["records"]
        iou = np.array([r["chosen_iou"] for r in recs])
        hit = np.array([r["loc_hit"] for r in recs], dtype=float)
        n = len(iou)
        bi = np.array([iou[rng.integers(0, n, n)].mean() for _ in range(B)])
        bh = np.array([hit[rng.integers(0, n, n)].mean() for _ in range(B)])
        lo_i, hi_i = np.percentile(bi, [2.5, 97.5])
        lo_h, hi_h = np.percentile(bh, [2.5, 97.5])
        out[f"{sys_name}/{scene}"] = {
            "n": n, "mIoU": float(iou.mean()), "ci95_iou": [float(lo_i), float(hi_i)],
            "mAcc": float(hit.mean()), "ci95_acc": [float(lo_h), float(hi_h)]}
        print(f"{sys_name:>6} {scene:<15} {iou.mean():.4f}   [{lo_i:.4f},{hi_i:.4f}]  "
              f"{hit.mean():.4f}   [{lo_h:.4f},{hi_h:.4f}]")

# per-system means with CI (mean of per-scene values, scene bootstrap approximated
# by averaging per-scene bootstrap draws)
print()
for sys_name in SYSTEMS:
    scenes = [s for s, _ in SYSTEMS[sys_name]]
    draws_i, draws_h = [], []
    for scene in scenes:
        d = out[f"{sys_name}/{scene}"]
        iou = np.array([r["chosen_iou"] for r in json.load(open(
            os.path.join(LSP if sys_name != "laga" else LAGA, dict(SYSTEMS[sys_name])[scene])))["records"]])
        hit = np.array([r["loc_hit"] for r in json.load(open(
            os.path.join(LSP if sys_name != "laga" else LAGA, dict(SYSTEMS[sys_name])[scene])))["records"]], dtype=float)
        n = len(iou)
        draws_i.append(np.array([iou[rng.integers(0, n, n)].mean() for _ in range(B)]))
        draws_h.append(np.array([hit[rng.integers(0, n, n)].mean() for _ in range(B)]))
    mi = np.mean(draws_i, axis=0)
    mh = np.mean(draws_h, axis=0)
    lo_i, hi_i = np.percentile(mi, [2.5, 97.5])
    lo_h, hi_h = np.percentile(mh, [2.5, 97.5])
    print(f"{sys_name:>6} MEAN  mIoU {mi.mean():.4f} [{lo_i:.4f},{hi_i:.4f}]   "
          f"mAcc {mh.mean():.4f} [{lo_h:.4f},{hi_h:.4f}]")

with open(os.path.join(LAGA, "bootstrap_ci.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote", os.path.join(LAGA, "bootstrap_ci.json"))
