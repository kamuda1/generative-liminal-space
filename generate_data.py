# todo: rename this file generate_masks?

import argparse
from skimage.io import imread
from skimage.segmentation import slic
import numpy as np
import random
from glob import glob
from PIL import Image
import os

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]

def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    img_size = canvas.shape[-1]
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas


def segment_image(image_path: str, n_segments: int) -> np.ndarray:
    """
    Segments an image.
    :param image_path:
        Path to the image
    :param n_segments:
        number of segments per image
    :return segments:
        The segments of the image
    """
    test_image = imread(image_path)
    segments = slic(test_image, n_segments=n_segments, compactness=100)
    return segments


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--image_dir', type=str, default='train_data')
    parser.add_argument('--save_dir', type=str, default='masks')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for filename in glob('/content/generative-liminal-space/train_data/*'):
    # for filename in glob('train_data/*'):
        masks = segment_image(filename, args.N)
        file_basename = os.path.basename(filename).split('.')[0]
        for mask_index in np.unique(masks):
            mask = (masks != mask_index).astype(float)
            img = Image.fromarray(mask * 255).convert('1')
            img.save('{:s}/{:s}_{:06d}.jpg'.format(args.save_dir, file_basename, mask_index))

    # for i in range(args.N):
    #     canvas = np.ones((args.image_size, args.image_size)).astype("i")
    #     ini_x = random.randint(0, args.image_size - 1)
    #     ini_y = random.randint(0, args.image_size - 1)
    #     mask = random_walk(canvas, ini_x, ini_y, args.image_size ** 2)
    #     print("save:", i, np.sum(mask))
    #
    #     img = Image.fromarray(mask * 255).convert('1')
    #     img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))
