import cv2
import numpy as np

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
    output_prefix: str = "output",
    num_disparities: int = 16 * 5,  # 必須是16的倍數
    block_size: int = 5,
    min_disparity: int = 0,
    disparity_threshold: int = 20,
    kernel: np.ndarray = np.ones((3, 3), np.uint8),  # 形態學運算的核
    invert_mask: bool = False  # 加入反轉遮罩的選項
):
    # 讀取左右影像（BGR格式）
    left_bgr = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None:
        print("無法讀取影像，請確認路徑正確。")
        return

    # 轉灰階並進行高斯模糊以減少雜訊
    left_gray = cv2.GaussianBlur(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    right_gray = cv2.GaussianBlur(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # 使用 StereoSGBM 計算視差
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

    # 將視差正規化至 0~255，便於觀察與後續處理
    disparity_norm = cv2.normalize(disparity_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_norm = disparity_norm.astype(np.uint8)

    # 建立遮罩：根據設定的閾值來區分前景與背景
    mask = disparity_norm > disparity_threshold
    if invert_mask:
        mask = np.logical_not(mask)

    # 進行形態學閉運算以平滑遮罩，避免小區域雜訊
    mask_morph = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)

    # 根據遮罩保留前景，將背景塗黑
    foreground = left_bgr.copy()
    foreground[mask_morph == 0] = [0, 0, 0]

    # 將前景取出後進行中心放大失真
    foreground = center_zoom_distortion(foreground, strength=0.5)
    background = center_shrink_distortion(left_bgr, strength=0.5)

    # 轉成灰階並建立遮罩
    gray_fg = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_fg, 10, 255, cv2.THRESH_BINARY)

    # 建立反向遮罩
    mask_inv = cv2.bitwise_not(mask)

    # 從背景中扣除要貼的區域
    bg_part = cv2.bitwise_and(background, background, mask=mask_inv)

    # 從前景圖取出要貼上的區域
    fg_part = cv2.bitwise_and(foreground, foreground, mask=mask)

    # 合併
    combined = cv2.add(bg_part, fg_part)


    # 顯示結果
    # cv2.imshow("Left Image", left_bgr)
    # cv2.imshow("Right Image", right_bgr)
    # cv2.imshow("Disparity", disparity_norm)
    # cv2.imshow("Mask", mask_morph * 255)
    # cv2.imshow("Foreground", foreground)
    # cv2.imshow("Background", background)
    cv2.imshow("Combined", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    num_disparities = 16 * 11
    block_size = 11

    main(
        left_path = "images/left.jpg",
        right_path = "images/right.jpg",
        output_prefix="result",
        num_disparities=num_disparities,
        block_size=block_size,
        min_disparity=0,
        disparity_threshold=35,
        invert_mask=False,
        kernel = np.ones((5, 5), np.uint8)
    )
    print(f"num_disparities: 16 * {int(num_disparities/16)}, block_size: {block_size}")
