from PIL import Image
import pandas as pd
import numpy as np
import open3d as o3d
import time


class point_cloud_generator():

    def __init__(self, rgb_file, depth_file, pc_file, f, scalingfactor):
        self.rgb_file = rgb_file                                # 彩色图片路径
        self.depth_file = depth_file                            # 深度图片路径
        self.pc_file = pc_file                                  # 存储点云路径
        self.f = f                                              # 焦距
        self.scalingfactor = scalingfactor                      # 深度信息的缩放因子，主要是单位换算
        self.rgb = Image.open(rgb_file)                         # 读取的彩色图
        self.depth = Image.open(depth_file).convert('I')        # 读取的深度图（单通道）
        self.width = self.rgb.size[0]                           # 图像的宽度
        self.height = self.rgb.size[1]                          # 图像的高度
        self.df = None                                          # 存储的3D点

    def calculate(self):
        t1 = time.time()
        depth = np.asarray(self.depth).T
        # 相机坐标系下 Zc = depth
        self.Z = depth / self.scalingfactor

        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))

        # 相机坐标系下 Xc = x * depth
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)
        # self.X = (X * self.Z) / self.f
        self.X = ((X - self.width / 2) * self.Z) / self.f

        # 相机坐标系下 Yc = y * depth
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        # self.Y = (Y * self.Z) / self.f
        self.Y = ((Y - self.height / 2) * self.Z) / self.f

        df = np.zeros((6, self.width * self.height))
        df[0] = self.X.T.reshape(-1)
        df[1] = self.Y.T.reshape(-1)
        df[2] = self.Z.T.reshape(-1)
        img = np.array(self.rgb)
        df[3] = img[:, :, 0].reshape(-1)
        df[4] = img[:, :, 1].reshape(-1)
        df[5] = img[:, :, 2].reshape(-1)
        self.df = df
        t2 = time.time()
        print('calcualte 3d point cloud Done.', t2 - t1)

    def write_ply(self):
        t1 = time.time()
        float_formatter = lambda x: "%.4f" % x
        points = []
        for i in self.df.T:
            # 设置 3D point 的属性值
            if i[3] == 0 and i[4] == 0 and i[5] == 0:
                continue
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(self.pc_file, "w")
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

        t2 = time.time()
        print("Write into .ply file Done.", t2 - t1)

    def show_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.pc_file)
        o3d.visualization.draw_geometries([pcd])


a = point_cloud_generator('../499b_mask.png', '../499b.png', '499b.ply',
                          f=1, scalingfactor=1)
a.calculate()
a.write_ply()
a.show_point_cloud()
df = a.df
np.save('pc.npy', df)

