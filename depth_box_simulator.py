import cv2
import numpy as np

# 視窗參數
W, H = 2000, 1200
FOCAL = 500.0
SPEED = 0.5      # 平移速度（IJ 快速點再調）
ANGLE_SPEED = 0.03  # 旋轉速度 (rad/frame)
BOX_COUNT = 20

# 隨機立方體
centers = np.column_stack([
    np.random.uniform(-5, 5, BOX_COUNT),
    np.random.uniform(-3, 3, BOX_COUNT),
    np.random.uniform( 2,12, BOX_COUNT)
])
sizes = np.random.uniform(0.5, 1.2, BOX_COUNT)

# 相機狀態
cam_pos = np.array([0.0, 0.0, 0.0])
yaw, pitch = 0.0, 0.0  # 偏航與俯仰

def rotation_matrix(yaw, pitch):
    """先繞 Y 軸 (yaw)，再繞 X 軸 (pitch) 的複合旋轉矩陣。"""
    Ry = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [          0, 1,          0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rx = np.array([
        [1,           0,            0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    return Rx @ Ry

def project(pt):
    """對 world 點做相機平移+旋轉，再投影到螢幕。"""
    # 平移
    p = pt - cam_pos
    # 旋轉到相機座標
    R = rotation_matrix(yaw, pitch)
    pc = R @ p
    x, y, z = pc
    if z <= 0.1:
        return None
    u = int(FOCAL * x / z + W/2)
    v = int(FOCAL * y / z + H/2)
    return (u, v), z

# cube 與 container 畫法不變，改用新的 project() 即可
# 這裡省略 draw_cube, draw_container 的程式，
# 直接沿用你上一版的實作，只要所有 project_point 都改成 project 就好

# 為簡潔示範，我先把 container 畫法也貼上：
def draw_container(img):
    corners = np.array([
        [+6, +4, 1],[+6, +4,15],[+6,-4, 1],[+6,-4,15],
        [-6, +4,1],[-6, +4,15],[-6,-4,1],[-6,-4,15],
    ])
    proj = []
    for c in corners:
        p = project(c)
        if not p: return
        (u,v),_ = p
        proj.append((u,v))
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
             (3,7),(4,5),(4,6),(5,7),(6,7)]
    for i,j in edges:
        cv2.line(img, proj[i], proj[j], (200,200,200),1,cv2.LINE_AA)

def draw_cube(img, center, size, color):
    dx,dy,dz = size, size, size
    corners = np.array([
        [+dx,+dy,+dz],[+dx,+dy,-dz],[+dx,-dy,+dz],[+dx,-dy,-dz],
        [-dx,+dy,+dz],[-dx,+dy,-dz],[-dx,-dy,+dz],[-dx,-dy,-dz],
    ]) + center
    proj,depths = [],[]
    for c in corners:
        p = project(c)
        if not p: return
        (u,v),z = p
        proj.append((u,v)); depths.append(z)
    proj = np.array(proj); depths = np.array(depths)
    idx = np.argsort(depths)
    front, back = idx[:4], idx[4:]
    hull_f = cv2.convexHull(proj[front].astype(np.int32))
    hull_b = cv2.convexHull(proj[back].astype(np.int32))
    col_b = tuple(int(c*0.6) for c in color)
    cv2.fillPoly(img, [hull_b], col_b, cv2.LINE_AA)
    cv2.fillPoly(img, [hull_f], color, cv2.LINE_AA)
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
             (3,7),(4,5),(4,6),(5,7),(6,7)]
    for i,j in edges:
        cv2.line(img, tuple(proj[i]), tuple(proj[j]), (0,0,0),1,cv2.LINE_AA)

cv2.namedWindow("裸視3D空間")

while True:
    canvas = np.zeros((H,W,3),dtype=np.uint8) + 30
    draw_container(canvas)
    order = np.argsort(centers[:,2])[::-1]
    for i in order:
        hue = int(180 * (i/BOX_COUNT))
        color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue,200,200]]]), cv2.COLOR_HSV2BGR)[0,0])
        draw_cube(canvas, centers[i], sizes[i], color)

    cv2.imshow("裸視3D空間", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

    # 平移（W/A/S/D）
    if key in (ord('w'),ord('W')): cam_pos[1] += SPEED
    if key in (ord('s'),ord('S')): cam_pos[1] -= SPEED
    if key in (ord('a'),ord('A')): cam_pos[0] -= SPEED
    if key in (ord('d'),ord('D')): cam_pos[0] += SPEED

    # **前後：依相機朝向**  
    # 先算出當前的旋轉矩陣
    R = rotation_matrix(yaw, pitch)
    # 相機 forward 向量 = R.T @ [0,0,1]
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    if key in (ord('q'),ord('Q')):
        cam_pos += forward * SPEED
    if key in (ord('e'),ord('E')):
        cam_pos -= forward * SPEED

    # 視角旋轉（I/J/K/L）
    if key in (ord('i'),ord('I')): pitch += ANGLE_SPEED
    if key in (ord('k'),ord('K')): pitch -= ANGLE_SPEED
    if key in (ord('j'),ord('J')): yaw   += ANGLE_SPEED
    if key in (ord('l'),ord('L')): yaw   -= ANGLE_SPEED

cv2.destroyAllWindows()

cv2.destroyAllWindows()
