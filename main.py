import cv2
import torch
import argparse
import numpy as np

from model_utils import *
from cam_utils import *


def parse():
    parser = argparse.ArgumentParser(description='set your identity parameters')
    parser.add_argument('--video_path', default='./data/groudline.avi', type=str)
    parser.add_argument('--cam_param_path', default='./data/camera_1.txt', type=str)
    parser.add_argument('--seg_weight_path', default='./deeplabv3/weights/model_best.pth.tar', type=str)
    parser.add_argument('--depth_encoder_path', default='./monodepth2/weights/encoder.pth', type=str)
    parser.add_argument('--depth_decoder_path', default='./monodepth2/weights/depth.pth', type=str)
    parser.add_argument('--point_save_path', default='./output/point/', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse()
    params = read_params(param_path=opt.cam_param_path)
    # ---------------- 初始化分割模型和深度模型 -------------
    seg_model = init_seg_net(opt.seg_weight_path, backbone='mobilenet', device=opt.device)
    depth_model, input_w, input_h = init_depth_net(opt.depth_encoder_path, opt.depth_decoder_path, opt.device)

    capture = cv2.VideoCapture(opt.video_path)
    i = 0
    while True:
        # ------- 依次读取每帧图片并处理 -------
        ret, frame = capture.read()
        if not ret:
            break
        img = Image.fromarray(frame.astype(np.uint8))
        # -------- 获取分割图与深度图 ----------
        # seg_img = generate_seg(seg_model, img, opt.device)      # 采用深度学习算法分割路面
        seg_img = seg_lane_by_cv(frame)                         # 采用opencv算法分割路面
        depth_img, disp_img = generate_depth(depth_model, input_h, input_w, img, opt.device)
        # depth_img = get_depth(seg_img, params)
        # --- 存储分割图和深度图，便于可视化对比 ----
        cv2.imwrite("./output/{}_rgb.png".format(i), frame)
        cv2.imwrite("./output/{}_depth.png".format(i), depth_img)
        cv2.imwrite("./output/{}_seg.png".format(i), seg_img)

        # ------ 根据深度图和分割图可视化点云 -------
        seg_img = frame
        pix2world(depth_img, seg_img, params, opt, i)
        i += 1

