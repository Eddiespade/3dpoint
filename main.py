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
    a = "W:" + str(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) + "  H:" + str(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) + \
        "  FPS:" + str(capture.get(cv2.CAP_PROP_FPS)) + "  TOTAL:" + str(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while True:
        # ------- 依次读取每帧图片并处理 -------
        ret, frame = capture.read()
        if not ret:
            break
        img = Image.fromarray(frame.astype(np.uint8))
        # -------- 获取分割图与深度图 ----------
        seg_img = generate_seg(seg_model, img, opt.device)
        depth_img, disp_img = generate_depth(depth_model, input_h, input_w, img, opt.device)
        # --- 存储分割图和深度图，便于可视化对比 ----
        # cv2.imwrite("./output/depth/{}.png".format(i), depth_img)
        # cv2.imwrite("./output/segment/{}.png".format(i), seg_img.transpose((1, 2, 0)))
        # cv2.imwrite("1_rgb.png", frame)
        # cv2.imwrite("1_depth.png", depth_img)

        # ------ 根据深度图和分割图可视化点云 -------
        # seg_img = frame.astype(np.float64).transpose(1, 2, 0)
        pix2world(depth_img, seg_img, params, opt, i)
        i += 1

        # cv2.putText(frame, a, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 200), 2)
        # print(frame.shape)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(10)
