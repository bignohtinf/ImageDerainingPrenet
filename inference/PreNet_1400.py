import torch
import cv2
import os
import numpy as np

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models.PReNet import PReNet

def load_model(weight_path, device):
    model = PReNet(recurrent_iter=6, use_GPU=(device=="cuda"))
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    return img

def postprocess(tensor):
    img = tensor.squeeze(0).clamp(0,1).cpu().numpy()
    img = img.transpose(1,2,0)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def derain_image(model, img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {img_path}")
        return

    inp = preprocess(img).to(next(model.parameters()).device)

    out, _ = model(inp)
    out = out[:, :, 4:-4, 4:-4]  # crop border

    result = postprocess(out)
    cv2.imwrite(save_path, result)

# -------------------------
# Derain folder
# -------------------------
def derain_folder(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"🔍 Found {len(images)} images")

    for name in images:
        inp_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)

        derain_image(model, inp_path, out_path)
        print(f"{name}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("output/checkpoints/prenet_100H.pth", device)

    derain_folder(
        model,
        input_dir="inference/test",
        output_dir="inference/output"
    )
