import imageio
import cv2
import numpy as np

from tqdm import tqdm

def center_zoom_distortion(img, strength=0.5):
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2

    # 建立每個像素的 (x, y) 座標網格
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)

    # 轉換為以中心為原點的座標
    dx = (xv - center_x) / center_x
    dy = (yv - center_y) / center_y
    radius = np.sqrt(dx**2 + dy**2)

    # 計算放大係數（radius 越小放大越多）
    scale = 1 + strength * (1 - radius)

    # 限制最大放大倍率避免超出邊界
    scale = np.clip(scale, 1, 1 + strength)

    # 映射回原圖座標
    map_x = center_x + dx * center_x / scale
    map_y = center_y + dy * center_y / scale

    # 轉為 float32 型別以供 remap 使用
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 套用重映射
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted


def center_shrink_distortion(img, strength=0.5):
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2

    # 建立每個像素的 (x, y) 座標網格
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)

    # 轉換為以中心為原點的座標
    dx = (xv - center_x) / center_x
    dy = (yv - center_y) / center_y
    radius = np.sqrt(dx**2 + dy**2)

    # 計算縮小倍率（radius 越小縮小越多）
    scale = 1 - strength * (1 - radius)

    # 限制最小縮放比例避免變形過度（例如避免 scale 變成負值）
    scale = np.clip(scale, 1 - strength, 1)

    # 映射回原圖座標
    map_x = center_x + dx * center_x / scale
    map_y = center_y + dy * center_y / scale

    # 轉為 float32 型別以供 remap 使用
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 套用重映射
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted


def main(
    left_path: str,
    right_path: str,
    output_gif_path: str = "output.gif",
    fg_strength_range=(0.0, 0.6),
    bg_strength_range=(0.0, 0.3),
    steps=20,
    num_disparities: int = 16 * 5,
    block_size: int = 5,
    min_disparity: int = 0,
    disparity_threshold: int = 20,
    kernel: np.ndarray = np.ones((3, 3), np.uint8),
    invert_mask: bool = False
    ):
    left_bgr = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None:
        print("❌ 無法讀取影像，請確認路徑正確。")
        return

    left_gray = cv2.GaussianBlur(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    right_gray = cv2.GaussianBlur(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity_raw[disparity_raw < 0] = 0
    disparity_norm = cv2.normalize(disparity_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    mask = disparity_norm > disparity_threshold
    if invert_mask:
        mask = np.logical_not(mask)
    mask_morph = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)

    frames = []

    fg_strengths = np.concatenate([
        np.linspace(fg_strength_range[0], fg_strength_range[1], steps),
        np.linspace(fg_strength_range[1], fg_strength_range[0], steps)
    ])
    bg_strengths = np.concatenate([
        np.linspace(bg_strength_range[0], bg_strength_range[1], steps),
        np.linspace(bg_strength_range[1], bg_strength_range[0], steps)
    ])

    print("📸 開始產生動畫幀...")
    for fg_strength, bg_strength in tqdm(zip(fg_strengths, bg_strengths), total=len(fg_strengths)):
        fg = left_bgr.copy()
        fg[mask_morph == 0] = [0, 0, 0]
        foreground = center_zoom_distortion(fg, strength=fg_strength)
        background = center_shrink_distortion(left_bgr, strength=bg_strength)

        gray_fg = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(gray_fg, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_bin)

        bg_part = cv2.bitwise_and(background, background, mask=mask_inv)
        fg_part = cv2.bitwise_and(foreground, foreground, mask=mask_bin)
        combined = cv2.add(bg_part, fg_part)

        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        frames.append(combined_rgb)

    print("💾 儲存 GIF...")
    imageio.mimsave(output_gif_path, frames, fps=10)
    print(f"✅ GIF 已儲存至：{output_gif_path}")

if __name__ == "__main__":
    main(
        left_path="images/left.jpg",
        right_path="images/right.jpg",
        output_gif_path="result/dolly_zoom.gif",
        fg_strength_range=(0.0, 0.3),
        bg_strength_range=(0.0, 0.5),
        steps=15,
        num_disparities=16 * 11,
        block_size=11,
        disparity_threshold=35,
        invert_mask=False,
        kernel=np.ones((5, 5), np.uint8)
    )

