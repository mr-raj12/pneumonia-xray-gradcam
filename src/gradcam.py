"""Grad-CAM (Selvaraju et al., 2017) implemented with forward/backward hooks."""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # Tensor hook avoids full-backward-hook issues with in-place activations.
        self.activations = out.detach()
        out.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1, 3, H, W). Returns (heatmap HxW in [0,1], probs, class_idx)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(dim=1)[0].detach()
        if class_idx is None:
            class_idx = int(probs.argmax())
        logits[0, class_idx].backward()

        # Channel weights = global-average-pooled gradients.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), probs.cpu().numpy(), class_idx

    def remove(self):
        self._handle.remove()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image: HxWx3 float in [0,1]; cam: HxW in [0,1]. Returns HxWx3 float blend."""
    heat = colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)
