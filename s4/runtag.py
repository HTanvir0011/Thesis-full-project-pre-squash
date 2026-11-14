"""Result-folder name of a kernels/down/run_down.py job, shared by the kernel and by ops/jobqueue.py (missing-run check), and the lookup
of a run in the outputs of earlier kernels (resume across Kaggle sessions, RUNPLAN_v3 R1)."""
import os


def tag(job):
    """job = [gpu, kind, data, variant, seed, epochs, *extra] -> e.g. nih_hyb_hybrid@densenet121_s0, bdcxr_cls_orig-soft_s1_f3,
    qp_xm_xraykd_s0, and nih_hyb_orig@swinv2_tiny_window8_256+xraykd_s0 for fine-tuning from that s4/xm.py run (--init)."""
    _, kind, name, variant, seed, _, *extra = job
    extra = [str(e) for e in extra]
    arch = extra[extra.index("--arch") + 1].split("/")[-1].split(".")[0] if "--arch" in extra else "densenet121"
    src = "+" + os.path.basename(os.path.dirname(extra[extra.index("--init") + 1])).split("_xm_")[-1].rsplit("_s", 1)[0] if "--init" in extra else ""
    t = f"{name}_{kind}_{variant.replace('+', '-')}" + (f"@{arch}{src}" if kind == "hyb" else "") + f"_s{seed}"
    if "--split" in extra:  # cross-validation fold (BDCXR): split_f<k>.npy -> suffix _f<k>
        t += "_" + extra[extra.index("--split") + 1].removeprefix("split_").removesuffix(".npy")
    return t


def previous_run(t, roots):
    """('done', dir) if an earlier kernel finished run t (result.json), ('resume', dir) if it left a checkpoint (ckpt.pt), else (None, None).
    roots: result folders of mounted earlier kernels, e.g. /kaggle/input/<slug>/results."""
    for r in roots:
        if os.path.exists(os.path.join(r, t, "result.json")):
            return "done", os.path.join(r, t)
    for r in roots:
        if os.path.exists(os.path.join(r, t, "ckpt.pt")):
            return "resume", os.path.join(r, t)
    return None, None
