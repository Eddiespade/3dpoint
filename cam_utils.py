import os
import cv2
import time
import numpy as np
import open3d as o3d


# ================================ 从txt中读取相机参数 ========================
def convert_format(s):
    length = len(s)
    if s[0] != '[' or s[length - 1] != ']':
        return float(s)
    s = np.asarray(s[1:length - 1].split(',')).astype(np.float64)
    return s


def read_params(param_path):
    with open(param_path, 'r') as f:
        data = f.read().strip().split('\n')
    params = {}
    for param in data:
        if param.strip() == '':
            continue
        param = param.split(":")
        params[param[0].strip()] = convert_format(param[1].strip())
    return params


# ================================ 根据平面假设获取地面深度 ========================
def get_depth(seg_path, cam_params, depth_path):
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
    cv2.imwrite(depth_path, depth_img)


# ================================ 根据 分割图和深度图 获取3D点云 ========================
def pix2world(depth_img, seg_img, cam_params, opt, idx):
    start = time.time()
    points = []

    # 获取图像的宽高
    h, w = depth_img.shape

    # 创建像素坐标系下的 (u, v, 1)，可以将numpy转为torch计算
    grid = np.meshgrid(range(w), range(h), indexing='xy')
    id_coord = np.stack(grid, axis=0).astype(np.float32)
    pix_coord = np.stack([id_coord[0].reshape(-1), id_coord[1].reshape(-1)], axis=0)
    ones = np.ones([w * h, ], dtype=np.float32)
    pix_points = np.stack([pix_coord[0], pix_coord[1], ones], axis=0)

    # ------- 图像坐标系 -> 相机坐标系 -------
    K = np.asarray(cam_params["K"]).reshape(3, -1)
    inv_K = np.linalg.inv(K)
    cam_points = inv_K @ pix_points
    cam_points = depth_img.reshape(1, -1) * cam_points  # ×Z

    # ------- 相机坐标系 -> 世界坐标系 -------
    P = np.asarray(cam_params["P"]).reshape(3, -1)
    T = P[:, 3]
    R = P[:, :3]
    inv_R = np.linalg.inv(R)
    world_point = inv_R @ (cam_points - T[:, np.newaxis])

    # ------- 根据分割图，获取图像颜色 -------
    seg_img = seg_img.reshape(3, -1)
    world_point = np.stack([world_point[0], world_point[1], world_point[2],
                            seg_img[0], seg_img[1], seg_img[2]], axis=0)

    # ------- 存储每个三维点，并过滤背景 --------
    float_format = lambda x: "%.4f" % x
    for p in world_point.T:
        # if p[3] == 0 and p[4] == 0 and p[5] == 0:
        #     continue
        # 设置 3D point 的属性值
        points.append("{} {} {} {} {} {} 0\n".format
                      (float_format(p[0]), float_format(p[1]), float_format(p[2]),
                       int(p[3]), int(p[4]), int(p[5])))
    if not os.path.exists(opt.point_save_path):
        os.mkdir(opt.point_save_path)
    point_path = opt.point_save_path + "/{}.ply".format(idx)
    file = open(point_path, "w")
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

    # ------- 显示3D点云 -------
    pcd = o3d.io.read_point_cloud(point_path)
    o3d.visualization.draw_geometries([pcd], window_name=str(idx), width=640, height=480)
    # visPcd(pcd, idx)
    return points


def visPcd(pcd, idx):  # 需要open3d,time库,默认暂停2秒，暂停时间在函数内设置
    # 创建可视化窗口并显示pcd
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(idx), width=640, height=480)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    # 设置窗口存在时间,根据需要自行更改
    time.sleep(1)
    # 关闭窗口
    vis.destroy_window()
