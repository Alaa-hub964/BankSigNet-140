"""
Difference GradCAM for MSFV-Net

Verification-specific explainability: backpropagates through the
absolute embedding deviation ||e_query - e_ref||_1 rather than
a classification score.

Also computes Deletion AUC and Insertion AUC faithfulness metrics
with mean-fill masking.

Reference:
    Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization, ICCV 2017.
    
    Samek et al., Evaluating the Visualization of What a Deep Neural
    Network Has Learned, IEEE TNNLS 2017.

Usage:
    python xai/difference_gradcam.py \
        --checkpoint verifier_msfv.pth \
        --query      path/to/query.png \
        --reference  path/to/ref.png \
        --output     gradcam_output.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.msfvnet import MSFVNet

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

DATASET_MEAN = 0.5  # mean-fill value for faithfulness evaluation


#  Difference GradCAM
def difference_gradcam(
    model: MSFVNet,
    query_img: torch.Tensor,
    ref_img: torch.Tensor,
    device: torch.device,
    target_layer_name: str = "encoder.features.7",
) -> np.ndarray:
    """
    Compute Difference GradCAM activation map.

    Gradient signal: S_diff = ||e_query - e_ref||_1

    Args:
        model          : MSFVNet in eval mode
        query_img      : (1, 1, 224, 224) query image tensor
        ref_img        : (1, 1, 224, 224) reference image tensor
        device         : torch device
        target_layer_name: name of layer to extract activations from

    Returns:
        cam: (224, 224) normalised activation map in [0, 1]
    """
    model.eval()

    activations = {}
    gradients = {}

    def fwd_hook(module, input, output):
        activations["value"] = output

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Register hooks on target layer
    target_layer = dict(model.named_modules())[target_layer_name]
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    query_img = query_img.to(device).requires_grad_(False)
    ref_img = ref_img.to(device).requires_grad_(False)

    # Forward — compute embedding deviation
    model.zero_grad()
    e_query = model.forward_one(query_img)
    with torch.no_grad():
        e_ref = model.forward_one(ref_img)

    # Gradient signal: L1 embedding deviation
    S_diff = (e_query - e_ref).abs().sum()
    S_diff.backward()

    h_fwd.remove()
    h_bwd.remove()

    # Grad-CAM weighting
    grads = gradients["value"]  # (1, C, H, W)
    acts = activations["value"]  # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalise to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam


# Faithfulness metrics
def deletion_insertion_auc(
    model: MSFVNet,
    query_img: torch.Tensor,
    ref_img: torch.Tensor,
    cam: np.ndarray,
    device: torch.device,
    n_steps: int = 10,
) -> dict:
    """
    Compute Deletion AUC and Insertion AUC with mean-fill masking.

    Args:
        model     : MSFVNet in eval mode
        query_img : (1, 1, 224, 224) query image tensor
        ref_img   : (1, 1, 224, 224) reference image tensor
        cam       : (224, 224) GradCAM activation map
        device    : torch device
        n_steps   : number of masking steps

    Returns:
        dict with 'deletion_auc', 'insertion_auc',
                  'deletion_scores', 'insertion_scores'
    """
    model.eval()
    query_np = query_img.squeeze().cpu().numpy()  # (224, 224)
    flat_cam = cam.flatten()
    sorted_idx = np.argsort(flat_cam)[::-1]  # descending importance

    with torch.no_grad():
        e_ref = model.forward_one(ref_img.to(device))

    def score(img_np):
        t = (
            torch.tensor(img_np, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        e = model.forward_one(t)
        return torch.sigmoid(model.forward_pair(e, e_ref)).item()

    n_pixels = len(flat_cam)
    del_scores, ins_scores = [], []

    for step in range(n_steps + 1):
        frac = step / n_steps
        n_mask = int(frac * n_pixels)

        # Deletion: replace top-n with mean
        del_img = query_np.copy().flatten()
        del_img[sorted_idx[:n_mask]] = DATASET_MEAN
        del_scores.append(score(del_img.reshape(224, 224)))

        # Insertion: reveal top-n, fill rest with mean
        ins_img = np.full_like(query_np.flatten(), DATASET_MEAN)
        ins_img[sorted_idx[:n_mask]] = query_np.flatten()[sorted_idx[:n_mask]]
        ins_scores.append(score(ins_img.reshape(224, 224)))

    steps = np.linspace(0, 1, n_steps + 1)
    del_auc = np.trapz(del_scores, steps)
    ins_auc = np.trapz(ins_scores, steps)

    return {
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "deletion_scores": del_scores,
        "insertion_scores": ins_scores,
        "steps": steps.tolist(),
    }


#  Visualisation
def overlay_cam(
    image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay GradCAM heatmap on image."""
    heatmap = cm.jet(cam)[:, :, :3]  # (H, W, 3) RGB
    img_rgb = np.stack([image_np] * 3, axis=-1)  # grayscale -> RGB
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
    overlay = alpha * heatmap + (1 - alpha) * img_rgb
    return (overlay * 255).astype(np.uint8)


#  Main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MSFVNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load images
    query_pil = Image.open(args.query).convert("L")
    ref_pil = Image.open(args.reference).convert("L")
    query_img = TRANSFORM(query_pil).unsqueeze(0)
    ref_img = TRANSFORM(ref_pil).unsqueeze(0)

    # Difference GradCAM
    cam = difference_gradcam(model, query_img, ref_img, device)

    # Faithfulness metrics
    faith = deletion_insertion_auc(model, query_img, ref_img, cam, device)
    print(f"Deletion AUC  : {faith['deletion_auc']:.4f}")
    print(f"Insertion AUC : {faith['insertion_auc']:.4f}")

    # Verification score
    with torch.no_grad():
        e_q = model.forward_one(query_img.to(device))
        e_r = model.forward_one(ref_img.to(device))
        prob = torch.sigmoid(model.forward_pair(e_q, e_r)).item()
    print(f"Genuine probability: {prob:.4f}")

    # Visualise
    query_np = query_img.squeeze().numpy()
    overlay = overlay_cam(query_np, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(query_np, cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Difference GradCAM")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (p={prob:.3f})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Difference GradCAM")
    parser.add_argument("--checkpoint", default="verifier_msfv.pth")
    parser.add_argument("--query", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default="gradcam_output.png")
    args = parser.parse_args()
    main(args)
