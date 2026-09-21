"""Small, explicit video-frame decoder for dataset smoke tests."""

from __future__ import annotations

from io import BytesIO
from math import isfinite
from subprocess import TimeoutExpired, run

from .libero_v3 import VideoFrameRef


class VideoDecodeError(RuntimeError):
    """Raised when an MP4 frame cannot be decoded as an RGB image."""


def decode_video_frame(
    reference: VideoFrameRef,
    *,
    expected_size: tuple[int, int] | None = None,
    timeout_seconds: float = 60.0,
):
    """Decode one timestamped frame with the system ``ffmpeg`` binary.

    ``-ss`` is placed after ``-i`` so the first implementation favors
    timestamp accuracy over random-access speed.  Batch decoding and caching
    should be added only after this contract is tested.
    """

    if reference.timestamp < 0 or not isfinite(reference.timestamp):
        raise VideoDecodeError("video timestamp must be finite and non-negative")
    if not reference.path.is_file():
        raise VideoDecodeError(f"video file does not exist: {reference.path}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(reference.path),
        "-ss",
        f"{reference.timestamp:.6f}",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    try:
        result = run(command, capture_output=True, check=False, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise VideoDecodeError("ffmpeg is required to decode LIBERO video frames") from exc
    except TimeoutExpired as exc:
        raise VideoDecodeError(f"ffmpeg timed out for {reference.path}") from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise VideoDecodeError(f"ffmpeg failed for {reference.path}: {detail}")

    try:
        from PIL import Image

        with Image.open(BytesIO(result.stdout)) as decoded:
            image = decoded.convert("RGB")
            image.load()
    except Exception as exc:  # Pillow exposes several decoder-specific errors.
        raise VideoDecodeError("ffmpeg output was not a readable image") from exc
    if expected_size is not None and image.size != expected_size:
        raise VideoDecodeError(f"expected image size {expected_size}, got {image.size}")
    return image
