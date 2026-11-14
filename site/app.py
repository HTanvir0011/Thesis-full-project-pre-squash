"""Hugging Face Space demo: upload a frontal chest radiograph, see the predicted bone image and the bone-suppressed image.
Research use only; not a medical device. Uploads are processed in memory and never stored or logged."""
import cv2
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from s4.infer import load_suppressor, prep512, suppress_batch, to_u8

REPO = "REPLACE_WITH_HF_MODEL_REPO"  # set when the model card and weights are published
CKPTS = [f"twin16_s{i}.pt" for i in range(3)]  # K = 3 deep ensemble; the bone image is the mean over the members
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOTICE = ("**Research prototype, not a medical device.** Do not use for diagnosis or any clinical decision. "
          "The model was trained only on synthetic radiographs rendered from CT scans; it has not been clinically validated. "
          "Your image is processed in memory and is not stored.")


def load():
    paths = [hf_hub_download(REPO, f) for f in CKPTS]
    assert not any("+" in p for p in paths), "load_suppressor splits on '+', so a path must not contain one"
    return load_suppressor("+".join(paths), DEVICE)


MODEL = load()


def run(img):
    if img is None:
        return None, None
    g = img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    x = prep512(g)
    b = suppress_batch(MODEL, x[None], DEVICE)[0]
    return to_u8(np.clip(b, 0, 1)[None], 512)[0], to_u8((x - b)[None], 512)[0]


# Gradio copies uploads into a temp folder; delete cached files after 60 s (checked every 60 s) so "not stored" holds, and switch telemetry off.
with gr.Blocks(title="Bone suppression (research demo)", analytics_enabled=False, delete_cache=(60, 60)) as demo:
    gr.Markdown("# Bone suppression in chest radiographs\n" + NOTICE)
    inp = gr.Image(type="numpy", label="Frontal chest radiograph", image_mode="L")
    with gr.Row():
        bone, soft = gr.Image(label="Predicted bone image"), gr.Image(label="Bone-suppressed image")
    gr.Button("Run").click(run, inp, [bone, soft])

if __name__ == "__main__":
    demo.launch()
