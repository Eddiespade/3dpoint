import os
import cv2
import sys
import numpy as np

camera_intrinsic = {
    # -------------- 外参 -------------------------
    "R": [[-0.91536173, 0.40180837, 0.02574754],
          [0.05154812, 0.18037357, -0.98224649],
          [-0.39931903, -0.89778361, -0.18581953]],
    "T": [1841.10702775, 4955.28462345, 1563.4453959],
    # -------------- 内参 -------------------------
    # 焦距，f/dx, f/dy
    "f": [1145.04940459, 1143.78109572],
    # principal point，主点，主轴与像平面的交点
    "c": [512.54150496, 515.45148698]

}


class CameraTools(object):

    @staticmethod
    def world2cam(point_world):
        """
        世界坐标系 -> 相机坐标系: R * pt + T:
        point_cam = np.dot(R, (point_world + T).T).T
        :return:
        """
        point_world = np.asarray(point_world)
        R = np.asarray(camera_intrinsic["R"])
        T = np.asarray(camera_intrinsic["T"])
        # 世界坐标系 -> 相机坐标系
        # np.dot() 和 @ 的作用一致，用于矩阵乘法
        point_cam = np.dot(R, point_world.T).T + T
        return point_cam

    @staticmethod
    def cam2world(point_world):
        """
        相机坐标系 -> 世界坐标系: inv(R) * pt - T
        point_cam = np.dot(inv(R), point_world.T).T - T
        :return:
        """
        point_world = np.asarray(point_world)
        R = np.asarray(camera_intrinsic["R"])
        T = np.asarray(camera_intrinsic["T"])
        # 相机坐标系 -> 世界坐标系
        point_cam = np.dot(np.linalg.inv(R), (point_world - T).T).T
        return point_cam

    @staticmethod
    def cam2pixel(cam_coord, f, c):
        """
        相机坐标系 -> 像素坐标系: (f / dx) * (X / Z) = f * (X / Z) / dx
        cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
        将从3D(X,Y,Z)映射到2D像素坐标P(u,v)计算公式为：
        u = X * fx / Z + cx
        v = Y * fy / Z + cy
        D(v,u) = Z / Alpha
        =====================================================
        camera_matrix = [[428.30114, 0.,   316.41648],
                        [   0.,    427.00564, 218.34591],
                        [   0.,      0.,    1.]])
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        =====================================================
        :param cam_coord:
        :param f: [fx,fy]
        :param c: [cx,cy]
        :return:
        """
        # 等价于：(f / dx) * (X / Z) = f * (X / Z) / dx
        # 三角变换， / dx, + center_x
        u = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
        v = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
        d = cam_coord[..., 2]
        return u, v, d

    @staticmethod
    def convert_cc_to_ic(point_cam):
        """
        相机坐标系 -> 像素坐标系
        :param point_cam:
        :return:
        """
        # 相机坐标系 -> 像素坐标系，并 get relative depth
        # Subtract center depth
        # 选择 Pelvis骨盆 所在位置作为相机中心，后面用之求relative depth
        root_idx = 0
        center_cam = point_cam[root_idx]  # (x,y,z) mm
        joint_num = len(point_cam)
        f = camera_intrinsic["f"]
        c = camera_intrinsic["c"]
        # joint image_dict，像素坐标系，Depth 为相对深度 mm
        joint_img = np.zeros((joint_num, 3))
        joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = CameraTools.cam2pixel(point_cam, f, c)  # x,y
        joint_img[:, 2] = joint_img[:, 2] - center_cam[2]  # z
        return joint_img


def demo_for_human36m():
    point_world = [[-91.679, 154.404, 907.261],
                   [-223.23566, 163.80551, 890.5342],
                   [-188.4703, 14.077106, 475.1688],
                   [-261.84055, 186.55286, 61.438915],
                   [39.877888, 145.00247, 923.98785],
                   [-11.675994, 160.89919, 484.39148],
                   [-51.550297, 220.14624, 35.834396],
                   [-132.34781, 215.73018, 1128.8396],
                   [-97.1674, 202.34435, 1383.1466],
                   [-112.97073, 127.96946, 1477.4457],
                   [-120.03289, 190.96477, 1573.4],
                   [25.895456, 192.35947, 1296.1571],
                   [107.10581, 116.050285, 1040.5062],
                   [129.8381, -48.024918, 850.94806],
                   [-230.36955, 203.17923, 1311.9639],
                   [-315.40536, 164.55284, 1049.1747],
                   [-350.77136, 43.442127, 831.3473],
                   [-102.237045, 197.76935, 1304.0605]]
    point_world = np.asarray(point_world)
    print(point_world.shape)
    # 关节点连接线
    # kps_lines = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
    #              (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    # # show in 世界坐标系
    # vis.vis_3d(point_world, kps_lines, coordinate="WC", title="WC", set_lim=True, isshow=True)
    #
    # kp_vis = CameraTools()
    #
    # # show in 相机坐标系
    # point_cam = kp_vis.convert_wc_to_cc(point_world)
    # vis.vis_3d(point_cam, kps_lines, coordinate="CC", title="CC", set_lim=True, isshow=True)
    # joint_img = kp_vis.convert_cc_to_ic(point_cam)
    #
    # point_world1 = kp_vis.convert_cc_to_wc(point_cam)
    # vis.vis_3d(point_world1, kps_lines, coordinate="WC", title="WC", set_lim=True, isshow=True)
    #
    # # show in 像素坐标系
    # kpt_2d = joint_img[:, 0:2]
    # image_path = "./data/s_01_act_02_subact_01_ca_02_000001.jpg"
    # image = image_processing.read_image(image_path)
    # image = image_processing.draw_key_point_in_image(image, key_points=[kpt_2d], pointline=kps_lines)
    # image_processing.cv_show_image("image_dict", image)


if __name__ == "__main__":
    demo_for_human36m()
