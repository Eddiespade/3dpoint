import os
import numpy as np
from PIL import Image
from monodepth2 import networks
from torchvision import transforms
from deeplabv3 import utils
from deeplabv3.modeling.deeplab import *
from torchvision.utils import make_grid


class FullModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(FullModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs[("disp", 0)]


# ============================= 初始化分割网络 ========================
def init_seg_net(ckpt, backbone, device, out_stride=16, classes=12):
    seg_model = DeepLab(num_classes=classes,
                        backbone=backbone,
                        output_stride=out_stride,
                        sync_bn=None,
                        freeze_bn=False)
    ckpt = torch.load(ckpt, map_location=device)
    seg_model.load_state_dict(ckpt['state_dict'])
    seg_model.to(device)
    seg_model.eval()
    return seg_model


# ============================= 初始化深度网络 =========================
def init_depth_net(ckpt_encoder, ckpt_decoder, device):
    encoder_path = ckpt_encoder
    depth_decoder_path = ckpt_decoder
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # 提取模型训练时使用的图像的高度和宽度
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    depth_model = FullModel(encoder, depth_decoder)
    depth_model.to(device)
    depth_model = depth_model.eval()
    return depth_model, feed_width, feed_height


# =============================== 生成分割图 ==========================
def generate_seg(net, img, device):
    composed_transforms = transforms.Compose([
        utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        utils.ToTensor()])
    tensor_in = composed_transforms(img).unsqueeze(0)
    tensor_in = tensor_in.to(device)

    with torch.no_grad():
        output = net(tensor_in)

    grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                           3, normalize=False, value_range=(0, 255))
    return grid_image.cpu().numpy()


# =============================== 生成深度图 ==========================
def generate_depth(net, feed_width, feed_height, img, device, scale=49):
    original_width, original_height = img.size
    image = img.resize((feed_width, feed_height), Image.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = net(image)

    disp = torch.nn.functional.interpolate(
        output, (original_height, original_width), mode="bilinear", align_corners=False)
    scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

    scaled_disp = scaled_disp.squeeze().cpu().numpy()
    scaled_disp = 10 * scaled_disp
    depth = depth.squeeze().cpu().numpy()
    depth = scale * depth
    return depth, scaled_disp


def disp_to_depth(disp, min_depth, max_depth):
    """
    将网络的 sigmoid 输出转换为深度预测，此转换的公式在论文的“其他注意事项”部分中给出。
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
