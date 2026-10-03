"""基础图像质量指标。

这个文件是作者自己的轻量指标封装，不是官方 SBFBurst 的评估脚本。
它主要服务早期传统/合成实验：输入一张参考图和一张重建图，返回
PSNR 与 SSIM 两个可以直接写入 JSON 的 Python float。

后面的 RealBSR/SBFBurst 正式实验会使用 ``real_data.py`` 中更完整的
``real_image_metrics``，以及官方/LPIPS 指标。
"""

from typing import TypedDict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .models import FloatImage


class ImageMetrics(TypedDict):
    """单张图像对比的最小指标集合。"""

    psnr: float
    ssim: float


def image_metrics(reference: FloatImage, candidate: FloatImage) -> ImageMetrics:
    """计算一对同尺寸灰度或 RGB 图像的 PSNR 和 SSIM。

    Args:
        reference: 参考图像，通常是真值图像，形状为 ``[H,W]`` 或
            ``[H,W,3]``，像素范围为 [0, 1]。
        candidate: 待评估图像，通常是算法输出，形状和 ``reference``
            相同，像素范围为 [0, 1]。

    Returns:
        ``{"psnr": ..., "ssim": ...}``。PSNR/SSIM 都是越高越好。
    """

    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    # 指标比较要求逐像素对齐，因此形状不一致时直接报错。
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must share a shape")
    if reference.ndim not in (2, 3):
        raise ValueError("images must have shape [H,W] or [H,W,3]")
    channel_axis = None
    if reference.ndim == 3:
        if reference.shape[-1] != 3:
            raise ValueError("RGB images must have exactly three channels")
        channel_axis = -1
    return {
        # data_range=1.0 表示输入已经归一化到 [0, 1]。
        "psnr": float(peak_signal_noise_ratio(reference, candidate, data_range=1.0)),
        "ssim": float(
            structural_similarity(
                reference,
                candidate,
                data_range=1.0,
                channel_axis=channel_axis,
            )
        ),
    }
