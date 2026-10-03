import numpy as np
import pytest

from robust_mfsr.metrics import image_metrics


def test_image_metrics_accepts_grayscale_images() -> None:
    reference = np.zeros((8, 8), dtype=np.float32)
    candidate = reference.copy()

    metrics = image_metrics(reference, candidate)

    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1.0)


def test_image_metrics_accepts_rgb_images() -> None:
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[3, 3, 1] = 0.25

    metrics = image_metrics(reference, candidate)

    assert np.isfinite(metrics["psnr"])
    assert 0.0 <= metrics["ssim"] < 1.0


def test_image_metrics_rejects_mismatched_shapes() -> None:
    reference = np.zeros((8, 8), dtype=np.float32)
    candidate = np.zeros((8, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="share a shape"):
        image_metrics(reference, candidate)


def test_image_metrics_rejects_non_rgb_channels() -> None:
    reference = np.zeros((8, 8, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="three channels"):
        image_metrics(reference, reference)
