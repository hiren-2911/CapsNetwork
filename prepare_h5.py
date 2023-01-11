import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y
import os

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    image_list =os.listdir(args.images_dir) #sorted(glob.glob('{}/*/*'.format(args.images_dir)))[:args.max_images]

    for i, image_path in enumerate(image_list):
        hr = pil_image.open(args.images_dir+image_path).convert('RGB')

        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        hr = np.clip(hr, 0.0, 255.0).astype(np.uint8)
        lr = np.clip(lr, 0.0, 255.0).astype(np.uint8)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        print(i)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        hr = np.clip(hr, 0.0, 255.0).astype(np.uint8)
        lr = np.clip(lr, 0.0, 255.0).astype(np.uint8)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir',default="/home/ml/Hiren/Data/Train/HR/")
    parser.add_argument('--output-path',default="/home/ml/Hiren/Data/Train/train.h5")
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=100)
    parser.add_argument('--max-images', type=int, default=50000)
    parser.add_argument('--eval',default=False ,action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
