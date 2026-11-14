"""BS-LDM (Sun et al., IEEE JBHI 2025; MIT licence, github.com/diaoquesang/BS-LDM) as an external real-pair-trained baseline.

Conditional latent diffusion: a VQGAN maps the 1024 px radiograph to a 4 x 128 x 128 latent, a UNet predicts the noise of the soft-tissue latent
conditioned on it, 1000 DDPM steps, then the VQGAN decoder gives the soft-tissue image. Trained on SZCH-X-Rays' real dual-energy TRAINING pairs (592),
so it is in-domain for the SZCH test pairs, whereas our models never see a real dual-energy pair.

The sampler follows the authors' ldm_eval.py: same scheduler settings, the clip range grows by 0.003 per step (scheduler rebuilt after every step),
the condition latent is re-attached after every step. Differences: initial and per-step noise come from a seeded generator (so a radiograph and its
lesion-inserted copy are sampled with identical noise), and an optional fp16 autocast. Weights are loaded only through s4.bsldm_safe.
"""
import numpy as np
import torch

from . import bsldm_safe

CFG = dict(num_train_timesteps=1000, beta_start=0.0008, beta_end=0.02, beta_schedule="squaredcos_cap_v2", clip_sample=True)
CLIP0, CLIP_RATE = 1.4, 0.003


class BSLDM:
    def __init__(self, device="cuda", fp16=False, steps=1000):
        bsldm_safe.setup_paths()
        w = bsldm_safe.THIRD / "bsldm_weights"
        self.dev, self.fp16, self.steps = device, fp16, steps
        self.vq = bsldm_safe.load(w / "2024-02-13-VQGAN.pth", device).eval()
        if not hasattr(self.vq, "use_checkpointing"):  # pickled with an older MONAI-generative: unpickling skips __init__, the newer class reads this flag
            self.vq.use_checkpointing = False
        self.net = bsldm_safe.load(w / "ldm-final-2024-02-16-myModel.pth", device).eval()

    @torch.no_grad()
    def soft(self, x01, seed=0, steps=None):
        """x01: [1024,1024] float in [0,1] (the radiograph) -> soft-tissue estimate [1024,1024] float32 on the authors' output scale (not clipped)."""
        from diffusers import DDPMScheduler
        steps = steps or self.steps
        mk = lambda r: DDPMScheduler(**CFG, clip_sample_range=r)  # noqa: E731
        ac = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16 and self.dev == "cuda")  # noqa: E731
        cxr = torch.from_numpy(np.ascontiguousarray(x01, np.float32))[None, None].to(self.dev) * 2 - 1  # their Normalize([0.5], [0.5])
        with ac():
            z = self.vq.encode_stage_2_inputs(cxr).float()
        g = torch.Generator(device=self.dev).manual_seed(seed)
        sample = torch.cat((torch.randn(z.shape, generator=g, device=self.dev), z), 1)
        sched = mk(CLIP0)
        ts = sched.timesteps
        if steps != len(ts):  # reduced-step variant (speed tests only): evenly spaced timesteps, the final step lands on t = 0
            ts = ts[torch.linspace(0, len(ts) - 1, steps).round().long()]
        for j, t in enumerate(ts):
            with ac():
                res = self.net(sample, torch.tensor([int(t)], device=self.dev).long()).float()
            sample = sched.step(res, t, sample, generator=g).prev_sample
            sched = mk(CLIP0 + CLIP_RATE * j)
            sample = torch.cat((sample[:, :4], z), 1)
        with ac():
            bs = self.vq.decode(sample[:, :4]).float()
        return (bs[0, 0] * 0.5 + 0.5).cpu().numpy()


class BoneFn:
    """Bone estimate for the lesion audit from a soft-tissue generator: B(x) = x - (a * S(x) + b), with (a, b) the least-squares fit of x on S(x) for the
    BASE image, reused for its lesion-inserted copies (they differ from the base in < 5% of the pixels), so only the generator's response to the lesion
    enters B(x') - B(x). Same noise seed for every call, so the base and perturbed samples share their noise."""

    def __init__(self, gen, seed=0):
        self.gen, self.seed, self.base, self.cal = gen, seed, None, None

    def __call__(self, x):
        s = self.gen.soft(x, self.seed)
        if self.base is None or self.base.shape != x.shape or (x != self.base).mean() > 0.05:  # a new radiograph (audit_real computes the base first)
            A = np.stack([s.ravel(), np.ones(s.size, np.float32)], 1)
            (a, b), *_ = np.linalg.lstsq(A, x.ravel().astype(np.float32), rcond=None)
            self.base, self.cal = x.copy(), (float(a), float(b))
        a, b = self.cal
        return (x - (a * s + b)).astype(np.float32)
