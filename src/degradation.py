"""生成可控的合成 burst，用于早期鲁棒 MFSR 实验。

这个文件是作者自己的实验脚手架，不是官方 SBFBurst 代码。它的目标是：
在进入 RealBSR/SBFBurst 之前，先用一张高分辨率图像构造一个“真值完全
可知”的多帧低分辨率 burst。

研究顺序里的作用：
1. 从一张干净 HR 图开始；
2. 通过平移、模糊、下采样、加噪声生成多帧 LR 图；
3. 把一部分非参考帧标成 outlier，并施加 blur/noise/occlusion；
4. 保存每帧真实 LR 位移、是否异常、异常区域 mask。

因为这些位移和异常标签是代码自己生成的，后续 registration、selection、
reliability 模块可以和已知真值对照。这是进入真实 RealBSR 实验前的
“小而可控”的验证阶段。
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import SynthesisConfig
from .models import FloatImage, FrameObservation


def _as_grayscale_float(image: NDArray[np.generic]) -> FloatImage:
    """把输入图像转成 [0, 1] 范围内的 float32 灰度图。

    合成生成器只在单通道上工作。如果输入是 RGB，就先转灰度；如果看起来
    是 0-255 的 8-bit 图，就归一化到 0-1。
    """

    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    if array.ndim != 2:
        raise ValueError("expected a grayscale or RGB image")
    output = array.astype(np.float32)
    if output.max(initial=0.0) > 1.0:
        output /= 255.0
    return np.clip(output, 0.0, 1.0)


def _shift(image: FloatImage, dx: float, dy: float) -> FloatImage:
    """用三次插值做亚像素平移。

    dx/dy 是像素单位的水平/垂直位移。边界使用反射填充，避免黑边影响
    后续误差分析。
    """

    height, width = image.shape
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    ).astype(np.float32)


def generate_synthetic_burst(
    high_resolution: NDArray[np.generic],
    config: SynthesisConfig = SynthesisConfig(),
) -> tuple[FloatImage, list[FrameObservation]]:
    """由一张高分辨率图生成一个低分辨率 burst。

    返回值包含裁剪后的 HR 真值，以及一组 ``FrameObservation``。每个
    observation 保存 LR 图、真实 LR 位移、是否 outlier、outlier mask。
    """

    # 先把源图归一化，并裁到能被 scale 整除的尺寸，保证 HR = scale * LR。
    hr = _as_grayscale_float(high_resolution)
    usable_height = hr.shape[0] - hr.shape[0] % config.scale
    usable_width = hr.shape[1] - hr.shape[1] % config.scale
    hr = hr[:usable_height, :usable_width]
    lr_size = (usable_width // config.scale, usable_height // config.scale)

    # 使用 config 中的随机种子，保证每次生成完全可复现。第 0 帧是参考帧，
    # 不参与 outlier 抽样。
    rng = np.random.default_rng(config.seed)
    candidate_indices = np.arange(1, config.frame_count)
    outlier_count = min(
        len(candidate_indices),
        int(round(config.frame_count * config.outlier_fraction)),
    )
    outlier_indices = set(
        rng.choice(candidate_indices, size=outlier_count, replace=False).tolist()
    )

    frames: list[FrameObservation] = []
    for index in range(config.frame_count):
        is_outlier = index in outlier_indices
        outlier_mask = np.zeros((lr_size[1], lr_size[0]), dtype=np.bool_)

        # 第 0 帧固定为参考帧。其他帧随机采样 LR 坐标系下的位移，再乘以
        # scale 作用到 HR 图上，模拟真实 burst 中的亚像素错位。
        if index == 0:
            dx_lr, dy_lr = 0.0, 0.0
        else:
            dx_lr, dy_lr = rng.uniform(
                -config.max_shift_lr,
                config.max_shift_lr,
                size=2,
            )

        # 在 HR 坐标系里施加几何平移。
        shifted = _shift(hr, dx_lr * config.scale, dy_lr * config.scale)

        # 每帧都有普通光学模糊；如果该帧是 blur/mixed outlier，则显著增强模糊。
        sigma = rng.uniform(config.blur_sigma_min, config.blur_sigma_max)
        if is_outlier and config.outlier_mode in {"blur", "mixed"}:
            sigma *= rng.uniform(2.8, 4.0)
        blurred = cv2.GaussianBlur(
            shifted,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT101,
        )

        # HR 下采样到 LR。INTER_AREA 是图像缩小时常用的局部平均型插值。
        low_resolution = cv2.resize(blurred, lr_size, interpolation=cv2.INTER_AREA)

        # 可选局部遮挡 outlier：随机矩形区域被随机纹理替换，并记录精确 mask。
        if is_outlier and config.outlier_mode in {"occlusion", "mixed"}:
            rect_width = max(4, lr_size[0] // 3)
            rect_height = max(4, lr_size[1] // 3)
            left = int(rng.integers(0, max(1, lr_size[0] - rect_width)))
            top = int(rng.integers(0, max(1, lr_size[1] - rect_height)))
            patch = rng.uniform(0.0, 1.0, size=(rect_height, rect_width))
            low_resolution[top : top + rect_height, left : left + rect_width] = patch
            outlier_mask[
                top : top + rect_height,
                left : left + rect_width,
            ] = True

        # 添加类似传感器噪声的高斯噪声。noise outlier 使用更大方差。
        # blur/noise 是整帧异常，因此 mask 标满整张图。
        noisy_outlier = is_outlier and config.outlier_mode in {"noise", "mixed"}
        noise_scale = config.noise_std * (5.0 if noisy_outlier else 1.0)
        if is_outlier and config.outlier_mode in {"blur", "noise"}:
            outlier_mask[:] = True
        low_resolution += rng.normal(0.0, noise_scale, size=low_resolution.shape)
        low_resolution = np.clip(low_resolution, 0.0, 1.0).astype(np.float32)

        # 保存图像和合成真值标签；后续模块可用这些字段检查估计是否正确。
        frames.append(
            FrameObservation(
                image=low_resolution,
                true_shift_lr=(float(dx_lr), float(dy_lr)),
                is_outlier=is_outlier,
                outlier_mask_lr=outlier_mask,
            )
        )

    return hr, frames
