"""SwinV2 encoder + nested (UNet++) decoder over the Swin stages + CBAM, with a light stride-4 -> 1 tail. Raw 1x1 head = bone image.
Simplification vs the reference (hardware-forced): SwinV2-T at 512 px instead of SwinV2-B at 1024 px; nesting only over the
Swin stages (strides 4-32), the stride-2/1 tail is a plain U-Net path using two shallow convolutional skips."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True), nn.Linear(max(c // r, 4), c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))[:, :, None, None]
        x = x * w
        return x * torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


def block(cin, cout, attn=True):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, CBAM(cout)) if attn else nn.Sequential(*layers)


def up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class Pyramid(nn.Module):
    """Plain ViT (one patch stride) -> maps at 1/4, 1/8, 1/16, 1/32 of the input by resampling four blocks' patch grids (shallow block = finest)."""

    def __init__(self, enc):
        super().__init__()
        self.enc, self.feature_info = enc, enc.feature_info

    def forward(self, x):
        H = x.shape[-1]
        return [F.interpolate(f, size=(H // s, H // s), mode="bilinear", align_corners=False) for f, s in zip(self.enc(x), (4, 8, 16, 32))]


class HFViT(nn.Module):
    """Hugging Face DINOv2-style ViT (e.g. microsoft/rad-dino, chest-radiograph pretrained) -> patch grids of four blocks, NCHW."""

    def __init__(self, name, pretrained, layers=(3, 6, 9, 12), size=518):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name) if pretrained else AutoModel.from_config(cfg)
        self.layers, self.size, self.g = layers, size, size // cfg.patch_size
        self.feature_info = type("FI", (), {"channels": staticmethod(lambda: [cfg.hidden_size] * 4)})()
        self._register_load_state_dict_pre_hook(self._attn_names)

    def _attn_names(self, state_dict, prefix, *_):
        """Checkpoints from an older transformers name the attention weights attention.attention.{query,key,value} and attention.output.dense, newer ones
        q_proj / k_proj / v_proj / o_proj: rename to whatever this version expects (same shapes, same maths)."""
        new = any(".q_proj." in n for n in self.state_dict())
        for old, nw in ((".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."),
                        (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj.")):
            src, dst = (old, nw) if new else (nw, old)
            for k in [k for k in state_dict if k.startswith(prefix) and src in k]:
                state_dict[k.replace(src, dst)] = state_dict.pop(k)

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[i][:, -self.g * self.g:].transpose(1, 2).reshape(x.shape[0], -1, self.g, self.g) for i in self.layers]  # drop cls/register tokens


def make_encoder(backbone, img_size, pretrained):
    if backbone.startswith("hf:"):
        return Pyramid(HFViT(backbone[3:], pretrained))
    if backbone.startswith("vit"):  # timm plain ViT, e.g. vit_small_patch16_dinov3.lvd1689m
        return Pyramid(timm.create_model(backbone, pretrained=pretrained, features_only=True, img_size=img_size, out_indices=(2, 5, 8, 11)))
    kw = dict(img_size=img_size) if "swin" in backbone else {}  # CNNs (e.g. convnextv2_tiny) take any input size
    return timm.create_model(backbone, pretrained=pretrained, features_only=True, **kw)


class SwinUNetPP(nn.Module):
    def __init__(self, backbone="swinv2_tiny_window8_256.ms_in1k", img_size=512, pretrained=True, ch=(48, 96, 192, 384)):
        super().__init__()
        self.enc = make_encoder(backbone, img_size, pretrained)
        enc_ch = self.enc.feature_info.channels()  # strides 4, 8, 16, 32
        self.register_buffer("mean", torch.tensor(0.45).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(0.225).view(1, 1, 1, 1))
        # 1x1 projections so every nested node works in the same small width per level
        self.proj = nn.ModuleList([nn.Conv2d(e, c, 1) for e, c in zip(enc_ch, ch)])
        n = {}  # nested nodes x[i][j]
        for j in range(1, 4):
            for i in range(0, 4 - j):
                n[f"{i}_{j}"] = block(ch[i] * j + ch[i + 1], ch[i])
        self.nodes = nn.ModuleDict(n)
        self.stem1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.stem2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.tail2 = block(ch[0] + 32, 32)
        self.tail1 = block(32 + 16, 16, attn=False)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):  # x: [B,1,H,W] in [0,1]
        s1 = self.stem1(x)
        s2 = self.stem2(s1)
        feats = self.enc(((x - self.mean) / self.std).expand(-1, 3, -1, -1))
        feats = [f.permute(0, 3, 1, 2) if f.shape[1] != p.in_channels else f for f, p in zip(feats, self.proj)]  # timm swin: NHWC
        e = [p(f) for p, f in zip(self.proj, feats)]
        X = {(i, 0): e[i] for i in range(4)}
        for j in range(1, 4):
            for i in range(0, 4 - j):
                inp = torch.cat([X[(i, k)] for k in range(j)] + [up(X[(i + 1, j - 1)], X[(i, 0)])], 1)
                X[(i, j)] = self.nodes[f"{i}_{j}"](inp)
        d = X[(0, 3)]
        d = self.tail2(torch.cat([up(d, s2), s2], 1))
        d = self.tail1(torch.cat([up(d, s1), s1], 1))
        return self.head(d)
