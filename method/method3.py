import os.path
import time

import cv2
import torch
import open3d as o3d
import numpy as np


def get_depth(seg_path, cam_params, depth_path):
    """根据相机高度h获得深度： z = h / (y - cy)
        1920*1080 耗时 8.9ms
    """
    seg_img = cv2.imread(seg_path)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

    h, w, _ = seg_img.shape
    grid = np.meshgrid(range(w), range(h), indexing='xy')
    id_coord = np.stack(grid, axis=0).astype(np.float32)
    pix_coord = np.stack([id_coord[0].reshape(-1), id_coord[1].reshape(-1)], axis=0)
    depth_img = np.zeros((w * h, 1))
    f = (cam_params["K"][0][0] + cam_params["K"][1][1]) / 2
    c_h = cam_params["H"]
    c_y = cam_params["K"][1][2]
    for i, p in enumerate(pix_coord.T):
        x, y = p[0], p[1]
        depth_img[i] = 100
        if y - c_y != 0:
            depth_img[i] = c_h * f / (y - c_y)
    depth_img = depth_img.reshape(h, w)
    cv2.imwrite(depth_path, depth_img)


def get_depth2(seg_path, cam_params, depth_path):
    """根据相机高度h获得深度： z = h / (y - cy)    加速版
        1920*1080 耗时 0.045ms
    """
    seg_img = cv2.imread(seg_path)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

    f = (cam_params["K"][0][0] + cam_params["K"][1][1]) / 2
    c_h = cam_params["H"]
    c_y = cam_params["K"][1][2]

    h, w, _ = seg_img.shape
    grid = np.meshgrid(range(w), range(h), indexing='xy')
    id_coord = np.stack(grid, axis=0).astype(np.float32)
    depth_img = c_h * f / (id_coord[1] - c_y)
    print(depth_img.shape)
    cv2.imwrite(depth_path, depth_img)


def float_format(x):
    return "%.4f" % x


def pix2world(depth_path, seg_path, cam_params, point_file):
    start = time.time()
    points = []
    # 读取深度图和分割图
    depth_img = cv2.imread(depth_path, 0)
    seg_img = cv2.imread(seg_path)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

    # 获取图像的宽高
    h, w = depth_img.shape

    # 创建像素坐标系下的 (u, v, 1)，可以将numpy转为torch计算
    grid = np.meshgrid(range(w), range(h), indexing='xy')
    id_coord = np.stack(grid, axis=0).astype(np.float32)
    pix_coord = np.stack([id_coord[0].reshape(-1), id_coord[1].reshape(-1)], axis=0)
    ones = np.ones([w * h, ], dtype=np.float32)
    pix_points = np.stack([pix_coord[0], pix_coord[1], ones], axis=0)

    # =============== 图像坐标系 -> 相机坐标系 ================
    K = np.asarray(cam_params["K"])
    inv_K = np.linalg.inv(K)
    cam_points = inv_K @ pix_points
    cam_points = depth_img.reshape(1, -1) * cam_points  # ×Z

    # =============== 相机坐标系 -> 世界坐标系 ================
    T = np.asarray(cam_params["T"])
    R = np.asarray(cam_params["R"])
    inv_R = np.linalg.inv(R)
    world_point = inv_R @ (cam_points - T[:, np.newaxis])

    # ================ 根据分割图，获取图像颜色 =================
    seg_img = seg_img.swapaxes(0, 2).swapaxes(1, 2).reshape(3, -1)
    world_point = np.stack([world_point[0], world_point[1], world_point[2],
                            seg_img[0], seg_img[1], seg_img[2]], axis=0)

    # =============== 存储每个三维点，并过滤背景 ==================
    for p in world_point.T:
        if p[3] == 0 and p[4] == 0 and p[5] == 0:
            continue
        # 设置 3D point 的属性值
        points.append("{} {} {} {} {} {} 0\n".format
                      (float_format(p[0]), float_format(p[1]), float_format(p[2]),
                       int(p[3]), int(p[4]), int(p[5])))

    file = open(point_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    end = time.time()
    print("获取3D点并存储分割点的时间为：%.4fms" % (end - start))

    # ===================== 显示3D点云 =========================
    pcd = o3d.io.read_point_cloud(point_file)
    o3d.visualization.draw_geometries([pcd])
    return points


if __name__ == '__main__':
    # 设置相机参数
    camera_parameters = {

        # # -------------- 外参 -------------------------
        # "R": [[1, 0, 0],
        #       [0, 1, 0],
        #       [0, 0, 1]],
        # "T": [0, 0, 0],
        # # -------------- 内参 -------------------------
        # "K": [[1, 0, 0],
        #       [0, 1, 0],
        #       [0, 0, 1]],
        # "H": 1000

        
        "R": [[-0.91536173, 0.40180837, 0.02574754],
              [0.05154812, 0.18037357, -0.98224649],
              [-0.39931903, -0.89778361, -0.18581953]],
        "T": [1841.10702775, 4955.28462345, 1563.4453959],
        "K": [[1145.04940459, 0, 512.54150496],
              [0, 1143.78109572, 515.45148698],
              [0, 0, 1]],
        "H": 1000
    }

    depth = '../test_img/499b.png'
    seg = '../499b_mask.png'
    # seg = '../499b.jpg'
    point = '../499b.ply'
    start = time.time()
    if not os.path.exists(depth):
        get_depth2(seg, camera_parameters, depth)
    end = time.time()
    print("获取深度图的时间为：%.4fms" % (end - start))
    point3D = pix2world(depth, seg, camera_parameters, point)
