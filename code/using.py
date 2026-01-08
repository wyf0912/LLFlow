import os
import argparse
import glob
import cv2
import numpy as np
import torch
from natsort import natsorted
from utils.util import opt_get
from models import create_model
from options import options as option


def fiFindByWildcard(wildcard):
    return natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path, model_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model.load_network(load_path=model_path, network=model.netG)
    model.netG = model.netG.cuda()  # Move model to GPU
    return model, opt


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t):
    return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach(
        ).cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255
    ).astype(np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result


def predict(model, opt, lr):
    # Pad image to ensure dimensions are compatible with the model
    h, w, c = lr.shape
    lr = impad(lr, bottom=int(np.ceil(h / 2) * 2 - h),
               right=int(np.ceil(w / 2) * 2 - w))

    lr_t = t(lr)
    lr_t = lr_t.cuda()  # Move to GPU

    # If log_low is enabled, apply log transform
    if opt["datasets"]["train"].get("log_low", False):
        lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))

    # If concat_histeq is enabled, concatenate histogram equalized image
    if opt.get("concat_histeq", False):
        # Apply histogram equalization to original image before padding
        his = hiseq_color_cv2_img(lr[:h, :w])  # Use original dimensions
        his = impad(his, bottom=int(np.ceil(h / 2) * 2 - h),
                    right=int(np.ceil(w / 2) * 2 - w))
        his_t = t(his).cuda()  # Move to GPU
        lr_t = torch.cat([lr_t, his_t], dim=1)

    # Use the same approach as test.py - get_sr method
    sr_t = model.get_sr(lq=lr_t, heat=None)
    return sr_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to the folder containing low-light images.")
    parser.add_argument("--output", required=True,
                        help="Path to the folder to save enhanced images.")
    parser.add_argument("--model_path", required=True,
                        help="Path to the trained model.")
    parser.add_argument("--conf", default="confs/Custom_smallNet_custom.yml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    model, opt = load_model(args.conf, args.model_path)

    input_dir = args.input
    output_dir = args.output

    lr_paths = fiFindByWildcard(os.path.join(input_dir, '*.*'))
    os.makedirs(output_dir, exist_ok=True)

    for lr_path in lr_paths:
        lr = imread(lr_path)
        h, w, c = lr.shape
        prediction = predict(model, opt, lr)
        prediction = rgb(prediction)

        # Crop to original size (remove padding)
        prediction = prediction[:h, :w]

        output_path = os.path.join(output_dir, os.path.basename(lr_path))
        imwrite(output_path, prediction)
        print(f"Processed: {os.path.basename(lr_path)}")

    print(f"Enhanced images saved to: {output_dir}")


if __name__ == "__main__":
    main()
