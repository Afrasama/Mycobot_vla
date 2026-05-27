import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p


@dataclass
class OverheadCameraConfig:
    """Top-down camera framing the MyCobot reachable workspace."""

    target_xyz: Tuple[float, float, float] = (0.22, 0.0, 0.02)
    distance_m: float = 0.62
    half_extent_m: float = 0.38
    up_vector: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    use_orthographic: bool = True
    fov_deg: float = 52.0
    near_val: float = 0.02
    far_val: float = 2.5
    width: int = 640
    height: int = 480
    min_body_pixels: int = 12


# Defaults frame robot base (0,0) + 0.20–0.30 m reach annulus with margin for the arm.
_OVERHEAD = OverheadCameraConfig(
    target_xyz=(
        float(os.getenv("OVERHEAD_TARGET_X", "0.22")),
        float(os.getenv("OVERHEAD_TARGET_Y", "0.0")),
        float(os.getenv("OVERHEAD_TARGET_Z", "0.02")),
    ),
    distance_m=float(os.getenv("OVERHEAD_DISTANCE_M", "0.62")),
    half_extent_m=float(os.getenv("OVERHEAD_HALF_EXTENT_M", "0.38")),
    use_orthographic=os.getenv("OVERHEAD_ORTHOGRAPHIC", "1") == "1",
    fov_deg=float(os.getenv("OVERHEAD_FOV_DEG", "52")),
    width=int(os.getenv("OVERHEAD_WIDTH", "640")),
    height=int(os.getenv("OVERHEAD_HEIGHT", "480")),
    min_body_pixels=int(os.getenv("OVERHEAD_MIN_BODY_PIXELS", "12")),
)


def get_overhead_config() -> OverheadCameraConfig:
    return _OVERHEAD


def configure_overhead_camera(
    *,
    target_xyz: Optional[Tuple[float, float, float]] = None,
    distance_m: Optional[float] = None,
    half_extent_m: Optional[float] = None,
    min_reach_m: Optional[float] = None,
    max_reach_m: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> OverheadCameraConfig:
    """
    Tune the overhead camera from workspace reach bounds.
    Centers on the reachable annulus in front of the robot base at (0, 0).
    """
    global _OVERHEAD
    if min_reach_m is not None and max_reach_m is not None:
        mid_r = 0.5 * (float(min_reach_m) + float(max_reach_m))
        half_extent_m = half_extent_m or (float(max_reach_m) + 0.10)
        target_xyz = target_xyz or (mid_r * 0.95, 0.0, 0.02)
    if target_xyz is not None:
        _OVERHEAD.target_xyz = tuple(float(v) for v in target_xyz)
    if distance_m is not None:
        _OVERHEAD.distance_m = float(distance_m)
    if half_extent_m is not None:
        _OVERHEAD.half_extent_m = float(half_extent_m)
    if width is not None:
        _OVERHEAD.width = int(width)
    if height is not None:
        _OVERHEAD.height = int(height)
    return _OVERHEAD


def get_overhead_eye_position(cfg: Optional[OverheadCameraConfig] = None) -> Tuple[float, float, float]:
    cfg = cfg or _OVERHEAD
    tx, ty, tz = cfg.target_xyz
    return (tx, ty, tz + cfg.distance_m)


def build_overhead_view_matrix(cfg: Optional[OverheadCameraConfig] = None) -> List[float]:
    cfg = cfg or _OVERHEAD
    eye = get_overhead_eye_position(cfg)
    return p.computeViewMatrix(
        cameraEyePosition=list(eye),
        cameraTargetPosition=list(cfg.target_xyz),
        cameraUpVector=list(cfg.up_vector),
    )


def _ortho_matrix(left, right, bottom, top, near, far) -> List[float]:
    """
    Column-major OpenGL orthographic projection matrix (16 floats).
    PyBullet older than ~3.2.6 lacks computeProjectionMatrixOrtho, so we
    build it manually instead.
    """
    rl = right - left
    tb = top - bottom
    fn = far - near
    return [
        2.0 / rl,          0.0,               0.0,              0.0,
        0.0,               2.0 / tb,          0.0,              0.0,
        0.0,               0.0,              -2.0 / fn,          0.0,
        -(right + left) / rl, -(top + bottom) / tb, -(far + near) / fn, 1.0,
    ]


def build_overhead_projection_matrix(
    cfg: Optional[OverheadCameraConfig] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[float]:
    cfg = cfg or _OVERHEAD
    w = int(width or cfg.width)
    h = int(height or cfg.height)
    if cfg.use_orthographic:
        ext = float(cfg.half_extent_m)
        # Try the native API first; fall back to manual matrix for older PyBullet builds.
        try:
            return p.computeProjectionMatrixOrtho(
                left=-ext, right=ext, bottom=-ext, top=ext,
                nearVal=cfg.near_val, farVal=cfg.far_val,
            )
        except AttributeError:
            return _ortho_matrix(-ext, ext, -ext, ext, cfg.near_val, cfg.far_val)
    return p.computeProjectionMatrixFOV(
        fov=cfg.fov_deg,
        aspect=w / max(h, 1),
        nearVal=cfg.near_val,
        farVal=cfg.far_val,
    )


def get_overhead_pixels_per_meter(
    width: Optional[int] = None,
    cfg: Optional[OverheadCameraConfig] = None,
) -> float:
    """Horizontal scale for world XY → overhead image pixels (orthographic)."""
    cfg = cfg or _OVERHEAD
    w = float(width or cfg.width)
    span_m = 2.0 * float(cfg.half_extent_m)
    return w / span_m if span_m > 1e-6 else 500.0


def _get_camera_image_safe(width, height, view_matrix, proj_matrix):
    last_exc = None
    for renderer in (p.ER_BULLET_HARDWARE_OPENGL, p.ER_TINY_RENDERER):
        try:
            return p.getCameraImage(
                width,
                height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=renderer,
            )
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Overhead camera capture failed: {last_exc}")


def get_overhead_camera_image(width=None, height=None):
    """
    Capture RGB + depth + segmentation from the fixed overhead task camera.
    Frames the robot base, arm motion, cube, and goal inside the workspace.
    """
    cfg = _OVERHEAD
    w = int(width or cfg.width)
    h = int(height or cfg.height)
    view_matrix = build_overhead_view_matrix(cfg)
    proj_matrix = build_overhead_projection_matrix(cfg, w, h)

    _, _, rgb_img, depth_img, seg_img = _get_camera_image_safe(
        w, h, view_matrix, proj_matrix
    )

    rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    depth_array = np.array(depth_img, dtype=np.float32).reshape(h, w)
    seg_array = np.array(seg_img, dtype=np.int32).reshape(h, w)
    return rgb_array, depth_array, seg_array


def sync_debug_workspace_camera():
    """
    PyBullet GUI camera: pitched 3/4 view so joint motion and depth are easy to see.
    Separate from the top-down task camera used for segmentation / pixel error.
    """
    cfg = _OVERHEAD
    tx, ty, tz = cfg.target_xyz
    target = [
        float(os.getenv("DEBUG_CAMERA_TARGET_X", str(tx))),
        float(os.getenv("DEBUG_CAMERA_TARGET_Y", str(ty))),
        float(os.getenv("DEBUG_CAMERA_TARGET_Z", "0.10")),
    ]
    p.resetDebugVisualizerCamera(
        cameraDistance=float(os.getenv("DEBUG_CAMERA_DISTANCE", "1.05")),
        cameraYaw=float(os.getenv("DEBUG_CAMERA_YAW", "52")),
        cameraPitch=float(os.getenv("DEBUG_CAMERA_PITCH", "-38")),
        cameraTargetPosition=target,
    )


def sync_debug_overhead_camera():
    """Backward-compatible alias — GUI now uses the workspace 3/4 camera."""
    sync_debug_workspace_camera()


def compute_body_centroid(segmentation_mask, body_id, min_pixels=None):
    cfg = _OVERHEAD
    min_px = int(min_pixels if min_pixels is not None else cfg.min_body_pixels)
    mask = segmentation_mask == body_id
    if int(np.sum(mask)) < min_px:
        return None
    ys, xs = np.where(mask)
    return float(np.mean(xs)), float(np.mean(ys))


def compute_cube_centroid(segmentation_mask, cube_id, min_pixels=None):
    return compute_body_centroid(segmentation_mask, cube_id, min_pixels=min_pixels)


def compute_pixel_error(centroid, width=640, height=480):
    center_x = width / 2
    center_y = height / 2
    return float(centroid[0] - center_x), float(centroid[1] - center_y)


def world_xy_to_overhead_pixel_error(dx_world, dy_world, scale=None):
    """
    Map world-frame XY offsets (target - reference) to overhead pixel error convention.
    Image Y increases downward; world +Y maps to decreasing image row.
    """
    if scale is None:
        scale = get_overhead_pixels_per_meter()
    return float(dx_world * scale), float(-dy_world * scale)


def overhead_visibility_report(
    segmentation_mask,
    body_ids: Dict[str, int],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Dict[str, object]:
    """Summarize which task bodies are visible in the overhead frame."""
    cfg = _OVERHEAD
    w = int(width or cfg.width)
    h = int(height or cfg.height)
    report: Dict[str, object] = {
        "image_size": (w, h),
        "pixels_per_meter": get_overhead_pixels_per_meter(w, cfg),
        "bodies": {},
        "all_visible": True,
    }
    for name, bid in body_ids.items():
        if bid is None or bid < 0:
            report["bodies"][name] = {"visible": False, "pixels": 0, "centroid": None}
            report["all_visible"] = False
            continue
        mask = segmentation_mask == bid
        px = int(np.sum(mask))
        centroid = compute_body_centroid(segmentation_mask, bid)
        visible = centroid is not None
        report["bodies"][name] = {
            "visible": visible,
            "pixels": px,
            "centroid": centroid,
        }
        if not visible:
            report["all_visible"] = False
    return report


def save_overhead_debug_frame(
    rgb: np.ndarray,
    path: str,
    segmentation_mask: Optional[np.ndarray] = None,
    body_ids: Optional[Dict[str, int]] = None,
) -> str:
    """Save an annotated overhead RGB frame for debugging / dashboard review."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Image = None

    if Image is None:
        return path

    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.line([(w // 2, 0), (w // 2, h)], fill=(80, 80, 80), width=1)
    draw.line([(0, h // 2), (w, h // 2)], fill=(80, 80, 80), width=1)

    colors = {
        "cube": (255, 80, 80),
        "gripper": (80, 220, 255),
        "robot": (80, 255, 120),
        "goal": (255, 200, 60),
    }
    if segmentation_mask is not None and body_ids:
        for name, bid in body_ids.items():
            c = compute_body_centroid(segmentation_mask, bid, min_pixels=4)
            if c is None:
                continue
            cx, cy = c
            col = colors.get(name, (255, 255, 255))
            r = 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)
            draw.text((cx + 8, cy - 8), name, fill=col)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path, format="PNG")
    return path


def capture_overhead_task_frame(
    body_ids: Dict[str, int],
    width: Optional[int] = None,
    height: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Capture RGB + segmentation and visibility diagnostics for the task scene."""
    rgb, _, seg = get_overhead_camera_image(width=width, height=height)
    report = overhead_visibility_report(seg, body_ids, width=width, height=height)
    if verbose:
        ppm = report["pixels_per_meter"]
        print(
            f"[OVERHEAD] {report['image_size'][0]}x{report['image_size'][1]} "
            f"ppm={ppm:.1f} all_visible={report['all_visible']}"
        )
        for name, info in report["bodies"].items():
            print(f"  {name}: visible={info['visible']} pixels={info['pixels']}")
    return rgb, seg, report


def get_relative_pixel_error_overhead(
    target_body_id,
    reference_body_id,
    width=640,
    height=480,
    verbose=False,
):
    """
    Pixel error of target relative to reference, using overhead camera.
    Returns (dx, dy) where:
      dx > 0 => target is to the RIGHT of reference
      dy > 0 => target is BELOW reference (image y axis points down)
    """
    _, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    target_centroid = compute_body_centroid(seg_mask, target_body_id)
    ref_centroid = compute_body_centroid(seg_mask, reference_body_id)

    if target_centroid is None or ref_centroid is None:
        if verbose:
            print("Target or reference not visible in overhead camera")
        return None

    dx = float(target_centroid[0] - ref_centroid[0])
    dy = float(target_centroid[1] - ref_centroid[1])

    if verbose:
        print("Target centroid:", target_centroid, "Ref centroid:", ref_centroid)
        print("Relative Pixel Error:", dx, dy)

    return dx, dy


def get_relative_pixel_error_overhead_and_rgb(
    target_body_id,
    reference_body_id,
    width=640,
    height=480,
    verbose=False,
):
    rgb, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    target_centroid = compute_body_centroid(seg_mask, target_body_id)
    ref_centroid = compute_body_centroid(seg_mask, reference_body_id)

    if target_centroid is None or ref_centroid is None:
        if verbose:
            print("Target or reference not visible in overhead camera")
        return None, rgb

    dx = float(target_centroid[0] - ref_centroid[0])
    dy = float(target_centroid[1] - ref_centroid[1])

    if verbose:
        print("Target centroid:", target_centroid, "Ref centroid:", ref_centroid)
        print("Relative Pixel Error:", dx, dy)

    return (dx, dy), rgb


def get_cube_pixel_error(robot, ee_index, cube_id, verbose=False):
    """
    Backwards-compatible helper.
    Returns cube pixel error relative to image center (overhead camera).
    """
    width = _OVERHEAD.width
    height = _OVERHEAD.height
    _, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    centroid = compute_cube_centroid(seg_mask, cube_id)
    if centroid is None:
        if verbose:
            print("Cube not visible in overhead camera")
        return None
    pixel_error_x, pixel_error_y = compute_pixel_error(
        centroid, width=width, height=height
    )
    if verbose:
        print("Centroid:", centroid)
        print("Pixel Error:", pixel_error_x, pixel_error_y)
    return pixel_error_x, pixel_error_y


def get_cube_pixel_error_and_rgb(robot, ee_index, cube_id, verbose=False):
    """
    Convenience helper: returns (pixel_error_x, pixel_error_y) OR None, plus RGB image.
    Overhead camera, relative to image center.
    """
    width = _OVERHEAD.width
    height = _OVERHEAD.height
    rgb, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    centroid = compute_cube_centroid(seg_mask, cube_id)
    if centroid is None:
        if verbose:
            print("Cube not visible in overhead camera")
        return None, rgb
    err = compute_pixel_error(centroid, width=width, height=height)
    if verbose:
        print("Centroid:", centroid)
        print("Pixel Error:", err[0], err[1])
    return err, rgb
