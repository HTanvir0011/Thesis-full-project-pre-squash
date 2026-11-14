import sys, numpy as np, cv2
sys.path.insert(0, "C:/Thesis new/swintrust4"); sys.path.insert(0, "C:/Thesis new/swintrust4/tools")
from s4 import eval_real as E
from szch_split import is_dev
meths = {"qure": E.make_qure("data/qure/weights/bone_suppression.ts", "cuda"), "ours_t16": E.make_ours("outputs/r3_dl/r3-3/runs/qp_twin16_s0_300k/final_ema.pt", "cuda"), "identity": lambda x: x}
bl = lambda z, s: cv2.GaussianBlur(z, (0, 0), s)
res = {}
for n, x, g in E.load_pairs("data/szch"):
    if not is_dev(n): continue
    for name, f in meths.items():
        est = f(x)
        for s in (0, 0.7, 1.0, 1.5):
            m = E.metrics(x, g, bl(est, s) if s else est)
            res.setdefault((name, s), []).append([m["ssim_cal"], m["psnr_cal"], m["psnr_cal_bone"], m["hp_ratio_nonbone"]])
print("method sigma ssim_cal psnr_cal psnr_bone hp_ratio (same output blur applied to every method, SZCH dev-24)")
for (k, s), v in res.items(): print(k.ljust(9), s, np.round(np.mean(v, 0), 4))
