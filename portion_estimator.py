# import torch
# import cv2
# import numpy as np

# # Load MiDaS (depth model)
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas.eval()

# transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = transforms.small_transform


# def estimate_portion(image):

#     img = np.array(image)
    
#     input_batch = transform(img)

#     with torch.no_grad():
#         depth = midas(input_batch)

#     depth = torch.nn.functional.interpolate(
#         depth.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

#     depth_map = depth.cpu().numpy()

#     # Simple threshold segmentation (food region)
#     mask = depth_map > np.percentile(depth_map, 60)

#     food_volume = np.sum(depth_map * mask)
#     total_volume = np.sum(depth_map)

#     portion_ratio = food_volume / total_volume

#     # Assume standard portion = 300g
#     estimated_grams = portion_ratio * 300

#     return round(estimated_grams, 2)


import numpy as np
import torch
import cv2
from rembg import remove, new_session

# =========================
# LOAD MODELS (CACHED)
# =========================

_midas = None
_transform = None
_rembg_session = None


def load_models():
    global _midas, _transform, _rembg_session

    if _midas is None:
        _midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        _midas.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        _transform = transforms.small_transform

    if _rembg_session is None:
        _rembg_session = new_session("u2netp")  # lightweight U²-Net

    return _midas, _transform, _rembg_session


# =========================
# PORTION ESTIMATION
# =========================
def estimate_portion(image):
    """
    Estimate portion size (grams) using:
    - U²-Net (rembg) → segmentation
    - MiDaS → depth
    """

    midas, transform_midas, rembg_session = load_models()

    # ⚡ Resize for speed
    image = image.resize((256, 256))
    img = np.array(image)

    # =========================
    # 1. SEGMENTATION (U²-Net)
    # =========================
    output = remove(img, session=rembg_session)
    output = np.array(output)

    if output.shape[-1] == 4:
        alpha = output[:, :, 3]
        food_mask = alpha > 0
    else:
        return 150.0  # fallback

    # Clean mask
    food_mask = food_mask.astype(np.uint8)
    food_mask = cv2.GaussianBlur(food_mask, (5, 5), 0)
    food_mask = (food_mask > 0.3).astype(np.uint8)

    if np.sum(food_mask) == 0:
        return 150.0

    # =========================
    # 2. DEPTH (MiDaS)
    # =========================
    input_batch = transform_midas(img)

    with torch.no_grad():
        depth = midas(input_batch)

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = depth.cpu().numpy()

    # Normalize depth
    depth_map = (depth_map - depth_map.min()) / (
        depth_map.max() - depth_map.min() + 1e-6
    )

    # =========================
    # 3. VOLUME ESTIMATION
    # =========================
    food_volume = np.sum(depth_map * food_mask)
    total_volume = np.sum(depth_map)

    if total_volume == 0:
        return 150.0

    ratio = food_volume / total_volume

    # =========================
    # 4. CONVERT TO GRAMS
    # =========================
    estimated_grams = ratio * 300  # base plate assumption

    # Clamp realistic range
    estimated_grams = np.clip(estimated_grams, 50, 500)

    return round(float(estimated_grams), 2)