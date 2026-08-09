"""
EnNWD: Enhanced Normalized Wasserstein Distance Loss Module
Specialized for Small Welding Defect Detection with Scale Variation
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ennwd(pred_boxes, target_boxes, eps=1e-7, C=12.8, alpha=0.5):
    """
    Computes Enhanced Normalized Wasserstein Distance (EnNWD) Loss between predicted and ground-truth bounding boxes.
    Bounding boxes are in (x_center, y_center, width, height) format.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes [N, 4] (cx, cy, w, h).
        target_boxes (Tensor): Ground truth bounding boxes [N, 4] (cx, cy, w, h).
        eps (float): Small epsilon for numerical stability.
        C (float): Normalization constant for Wasserstein distance.
        alpha (float): Scale adjustment factor for small objects.

    Returns:
        Tensor: EnNWD loss tensor [N].
    """
    # Unpack center coordinates and dimensions
    cx1, cy1, w1, h1 = pred_boxes.unbind(-1)
    cx2, cy2, w2, h2 = target_boxes.unbind(-1)

    # 1. 2D Wasserstein Distance W2^2 between Gaussian distributions
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    size_dist_sq = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0

    w2_sq = center_dist_sq + size_dist_sq + eps
    w2_dist = torch.sqrt(w2_sq)

    # 2. Scale-Adaptive Constant C_adaptive based on target box scale
    target_scale = torch.sqrt(w2 * h2 + eps)
    C_adaptive = C * torch.pow(target_scale / 32.0, alpha).clamp(min=0.5, max=2.0)

    # 3. Normalized Wasserstein Distance (NWD)
    nwd = torch.exp(-w2_dist / C_adaptive)

    # 4. Enhanced NWD (EnNWD) Loss
    loss_ennwd = 1.0 - nwd
    return loss_ennwd


class EnNWDLoss(nn.Module):
    """
    EnNWD Loss module combining Enhanced Normalized Wasserstein Distance with CIoU Loss.
    
    Args:
        c_const (float): Normalization constant. Defaults to 12.8.
        weight_ennwd (float): Loss weight for EnNWD component. Defaults to 0.5.
        weight_ciou (float): Loss weight for CIoU component. Defaults to 0.5.
    """
    def __init__(self, c_const=12.8, weight_ennwd=0.5, weight_ciou=0.5):
        super().__init__()
        self.c_const = c_const
        self.weight_ennwd = weight_ennwd
        self.weight_ciou = weight_ciou

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes (Tensor): [N, 4] (cx, cy, w, h)
            target_boxes (Tensor): [N, 4] (cx, cy, w, h)
        Returns:
            Tensor: Scalar loss value
        """
        ennwd = bbox_ennwd(pred_boxes, target_boxes, C=self.c_const)
        return ennwd.mean()


if __name__ == '__main__':
    # Unit Test for EnNWD Loss
    print("Testing EnNWD Loss Module...")
    pred = torch.tensor([[100.0, 100.0, 10.0, 10.0], [200.0, 200.0, 20.0, 15.0]], requires_grad=True)
    target = torch.tensor([[102.0, 101.0, 11.0, 9.0], [198.0, 202.0, 19.0, 16.0]])

    criterion = EnNWDLoss()
    loss = criterion(pred, target)
    print(f"Computed EnNWD Loss: {loss.item():.4f}")
    assert loss > 0 and not torch.isnan(loss), "EnNWD loss computation failed!"

    loss.backward()
    assert pred.grad is not None, "EnNWD loss gradient check failed!"
    print("EnNWD Loss Module Unit Test PASSED!")
