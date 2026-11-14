"""Safe loading of BS-LDM's released weights (third-party full-model pickles: torch.load(...) returns the whole nn.Module).

Pickles can run code while loading, so: (1) `scan` lists every global the pickle references WITHOUT executing it; (2) `load` unpickles with an allowlist
(`ALLOWED`) that holds only tensor-rebuild helpers, torch.nn layer classes, plain containers and the BS-LDM / MONAI-generative model classes, and
raises on anything else (os.system, builtins.eval, subprocess, ...). Reviewed 2026-10-31 against the observed globals of the two SZCH files.

python -I s4/bsldm_safe.py scan  <file.pth>            # prints the referenced globals and flags those outside the allowlist
python -I s4/bsldm_safe.py load  <file.pth> <out.pt>   # safe load, saves the state_dict (tensors only) for later weights_only=True use
BSLDM_HOME (default <repo>/../third_party) holds bsldm_src/ (the repo's .py files), pylibs/ (monai, monai-generative, diffusers) and bsldm_weights/.
"""
import io
import os
import pickle
import pickletools
import re
import sys
import zipfile
from pathlib import Path

THIRD = Path(os.environ.get("BSLDM_HOME", Path(__file__).resolve().parents[2] / "third_party"))  # code and libs of BS-LDM live apart from our repo
# Exact allowlist = the globals observed in the two SZCH files (VQGAN, UNet), reviewed one by one on 2026-10-31: tensor-rebuild helpers, torch.nn layer
# classes, the repo's own model classes (modules.*, MIT) and the MONAI-generative VQ-VAE classes. Nothing that touches files, processes or the network.
ALLOWED_EXACT = {
    ("collections", "OrderedDict"), ("builtins", "set"), ("__builtin__", "set"),
    ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"), ("torch", "FloatStorage"), ("torch", "float32"),
    ("torch.nn.modules.activation", "ReLU"), ("torch.nn.modules.activation", "SiLU"), ("torch.nn.modules.container", "ModuleList"),
    ("torch.nn.modules.container", "Sequential"), ("torch.nn.modules.conv", "Conv1d"), ("torch.nn.modules.conv", "Conv2d"),
    ("torch.nn.modules.conv", "ConvTranspose2d"), ("torch.nn.modules.dropout", "Dropout"), ("torch.nn.modules.linear", "Identity"),
    ("torch.nn.modules.linear", "Linear"), ("torch.nn.modules.pooling", "AvgPool2d"), ("torch.nn.modules.sparse", "Embedding"),
    ("modules.nn", "GroupNorm32"), ("modules.unet", "AttentionBlock"), ("modules.unet", "Downsample"), ("modules.unet", "QKVAttentionLegacy"),
    ("modules.unet", "ResBlock"), ("modules.unet", "TimestepEmbedSequential"), ("modules.unet", "UNetModel"), ("modules.unet", "Upsample"),
    ("generative.networks.layers.vector_quantizer", "EMAQuantizer"), ("generative.networks.layers.vector_quantizer", "VectorQuantizer"),
    ("generative.networks.nets.vqvae", "Decoder"), ("generative.networks.nets.vqvae", "Encoder"), ("generative.networks.nets.vqvae", "VQVAE"),
    ("generative.networks.nets.vqvae", "VQVAEResidualUnit"), ("monai.networks.blocks.acti_norm", "ADN"), ("monai.networks.blocks.convolutions", "Convolution"),
}


def allowed(mod, name):
    return (mod, name) in ALLOWED_EXACT


def pickle_bytes(path):
    """-> the pickle stream(s) of a torch file: data.pkl inside the zip (new format) or the file itself (legacy)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            return [z.read(n) for n in z.namelist() if n.endswith("data.pkl")]
    return [Path(path).read_bytes()]


def globals_of(buf):
    """Every (module, name) a pickle stream refers to, found by walking the opcodes (nothing is executed)."""
    out, strings, memo = set(), [], {}
    for op, arg, _ in pickletools.genops(io.BytesIO(buf)):
        n = op.name
        if n in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(arg)
        elif n in ("BINGET", "LONG_BINGET", "GET"):
            strings.append(memo.get(arg))
        elif n in ("MEMOIZE",):
            memo[len(memo)] = strings[-1] if strings else None
        elif n in ("BINPUT", "LONG_BINPUT", "PUT"):
            memo[arg] = strings[-1] if strings else None
        elif n == "GLOBAL":
            out.add(tuple(arg.split(" ", 1)))
        elif n == "STACK_GLOBAL":
            out.add((strings[-2], strings[-1]))
        elif n in ("REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX", "INST", "OBJ"):
            pass
    return out


def scan(path):
    found = set()
    for buf in pickle_bytes(path):
        found |= globals_of(buf)
    bad = sorted(g for g in found if not allowed(*g))
    return sorted(found), bad


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if not allowed(module, name):
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


class safe_pickle:  # minimal stand-in for the `pickle` module that torch.load accepts via pickle_module=
    Unpickler = SafeUnpickler
    UnpicklingError = pickle.UnpicklingError
    load = staticmethod(lambda f, **kw: SafeUnpickler(f, **kw).load())
    dumps, Pickler = staticmethod(pickle.dumps), pickle.Pickler


def setup_paths():
    """BS-LDM code (modules/) and the isolated MONAI / diffusers install; never put these on the main environment."""
    for p in (THIRD / "pylibs", THIRD / "bsldm_src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load(path, device="cpu"):
    import torch
    setup_paths()
    found, bad = scan(path)
    if bad:
        raise SystemExit(f"refusing to load {path}: globals outside the allowlist: {bad}")
    return torch.load(path, map_location=device, pickle_module=safe_pickle, weights_only=False)


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "scan":
        found, bad = scan(path)
        print(f"{path}: {len(found)} distinct globals")
        for g in found:
            print("  ", "BLOCK" if g in bad else "ok   ", ".".join(g))
        sys.exit(1 if bad else 0)
    if cmd == "load":
        import torch
        m = load(path)
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save(sd, sys.argv[3])
        print(type(m).__module__ + "." + type(m).__name__, f"{sum(v.numel() for v in sd.values()) / 1e6:.1f}M parameters -> {sys.argv[3]}")
