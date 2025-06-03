import imageio
import cv2
import numpy as np
import os
from tqdm import tqdm

def recut(img, strength=0.5):
    """
    將影像等比例剪切後再放大到原本大小，
    等同在畫面中心做一個 zoom‐in（以 strength 控制 zoom 程度）。
    """
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2

    # 計算縮小後的尺寸
    shrink_h = int(h * (1 - strength))
    shrink_w = int(w * (1 - strength))

    # 計算裁剪區域的起始位置
    start_x = (w - shrink_w) // 2
    start_y = (h - shrink_h) // 2

    # 裁剪圖片（以畫面中心做縮小）
    cropped = img[start_y : start_y + shrink_h, start_x : start_x + shrink_w]

    # 放大裁剪後的圖片至原圖大小
    distorted = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return distorted


def center_zoom_distortion(img, strength=0.5):
    """
    以畫面中心為原點做「中心放大扭曲」（dolly‐zoom 效果的前景動作）。
    strength 越大，越往中心收放時的扭曲越明顯。
    """
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2

    # 建立每個像素的 (x, y) 座標網格
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)

    # 轉換為以中心為原點的歸一化座標
    dx = (xv - center_x) / center_x
    dy = (yv - center_y) / center_y
    radius = np.sqrt(dx**2 + dy**2)

    # 計算放大係數（radius 越小放大越多）
    # 這邊 scale 在 [1, 1+strength] 之間變動
    scale = 1 + strength * (1 - radius)
    scale = np.clip(scale, 1, 1 + strength)

    # 映射回原圖座標
    map_x = center_x + dx * center_x / scale
    map_y = center_y + dy * center_y / scale

    # 轉成 float32 以供 remap 使用
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 套用重映射
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted


def center_shrink_distortion(img, strength=0.5):
    """
    以畫面中心為原點做「中心縮小扭曲」（dolly‐zoom 效果的背景動作）。
    strength 越大，越往中心收縮越明顯。
    """
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2

    # 建立每個像素的 (x, y) 座標網格
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x, y)

    # 轉換為以中心為原點的歸一化座標
    dx = (xv - center_x) / center_x
    dy = (yv - center_y) / center_y
    radius = np.sqrt(dx**2 + dy**2)

    # 計算縮小倍率（radius 越小縮小越多）
    # 這邊 scale 在 [1–strength, 1] 之間
    scale = 1 - strength * (1 - radius)
    scale = np.clip(scale, 1 - strength, 1)

    # 映射回原圖座標
    map_x = center_x + dx * center_x / scale
    map_y = center_y + dy * center_y / scale

    # 轉成 float32 以供 remap 使用
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 套用重映射
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return distorted


def main(
    color_path: str,
    depth_path: str,
    output_gif_path: str = "output.gif",
    fg_strength_range=(0.0, 0.6),   # 前景 zoom‐in 的強度範圍
    bg_strength_range=(0.0, 0.3),   # 背景 shrink‐out 的強度範圍
    steps: int = 20,                # 動畫總共幾個幀
    disparity_threshold: int = 20,  # 從深度圖做二值化時的閾值（0–255）
    kernel: np.ndarray = np.ones((3, 3), np.uint8),
    invert_mask: bool = False       # 如果想反轉前景/背景遮罩，設定 True
):
    """
    1) 讀取彩色影像 (color_path)
    2) 讀取深度影像 (depth_path)，假設已經是 0–255 的灰階
    3) 以 depth > disparity_threshold 做二值遮罩 → mask
       (如果 invert_mask=True，則反相)
    4) 對遮罩做 morphology close → mask_morph
    5) 根據 mask_morph 分離前景/背景，交叉應用 center_zoom_distortion / center_shrink_distortion
    6) 繼續做 recut（center 裁剪再放大）增加邊緣效果
    7) 合成各幀，存成 GIF
    """

    # 讀取彩色圖（作為原圖＆遮罩貼用）
    color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        print(f"❌ 無法讀取彩色影像：{color_path}，請確認路徑是否正確。")
        return

    # 讀取深度圖 (灰階)，假設已經經過正規化到 0–255
    depth_gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth_gray is None:
        print(f"❌ 無法讀取深度影像：{depth_path}，請確認路徑是否正確。")
        return

    # 以 depth > threshold 做前景遮罩
    mask = depth_gray > disparity_threshold
    if invert_mask:
        mask = np.logical_not(mask)

    # morphology close 讓遮罩邊緣更平滑
    mask_morph = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)

    # 準備前景/背景強度線性插值
    fg_strengths = np.concatenate([
        np.linspace(fg_strength_range[0], fg_strength_range[1], steps),
        np.linspace(fg_strength_range[1], fg_strength_range[0], steps)
    ])
    bg_strengths = np.concatenate([
        np.linspace(bg_strength_range[0], bg_strength_range[1], steps),
        np.linspace(bg_strength_range[1], bg_strength_range[0], steps)
    ])

    frames = []
    print("📸 開始產生 dolly‐zoom 動畫幀...")

    for fg_s, bg_s in tqdm(zip(fg_strengths, bg_strengths), total=len(fg_strengths)):
        # 1. 前景：把原圖上 mask_morph==0 的部分抹掉
        fg = color_bgr.copy()
        fg[mask_morph == 0] = [0, 0, 0]

        # 2. foreground 做中心放大扭曲
        foreground = center_zoom_distortion(fg, strength=fg_s)
        foreground = recut(foreground, strength=0.3)

        # 3. 背景：直接對整張原圖做中心縮小扭曲
        background = center_shrink_distortion(color_bgr, strength=bg_s)
        background = recut(background, strength=0.3)

        # 4. 生成一個二值遮罩，用於合成最終畫面
        gray_fg = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(gray_fg, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_bin)

        # 5. 合成：背景部分 + 前景部分
        bg_part = cv2.bitwise_and(background, background, mask=mask_inv)
        fg_part = cv2.bitwise_and(foreground, foreground, mask=mask_bin)
        combined = cv2.add(bg_part, fg_part)

        # 6. 轉成 RGB，放進 frames
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        # 7. 高斯濾波去除雜訊（可選）
        combined_rgb = cv2.GaussianBlur(combined_rgb, (11, 11), 0)

        frames.append(combined_rgb)

    # 確保輸出資料夾存在
    output_dir = os.path.dirname(output_gif_path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 存 GIF
    print("💾 開始儲存 GIF...")
    imageio.mimsave(output_gif_path, frames, fps=10)
    print(f"✅ GIF 已儲存至：{output_gif_path}")


if __name__ == "__main__":
    # 範例呼叫：請把 color.jpg、depth.jpg 放在 script 同一個目錄，或自己改成正確路徑
    main(
        color_path="images/color.jpg",
        depth_path="images/depth.jpg",
        output_gif_path="result/dolly_zoom.gif",
        fg_strength_range=(0.0, 0.2),
        bg_strength_range=(0.0, 0.2),
        steps=40,
        disparity_threshold=170,
        invert_mask=False,
        kernel=np.ones((5, 5), np.uint8)
    )
