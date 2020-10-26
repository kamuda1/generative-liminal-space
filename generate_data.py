# todo: rename this file generate_masks?

import argparse
from skimage.io import imread
from skimage.segmentation import slic
from skimage.transform import resize
import numpy as np
from glob import glob
from PIL import Image
import os
import matplotlib.pyplot as plt


def segment_image(image_path: str, n_segments: int) -> np.ndarray:
    """
    Segments an image. Rescales the image to 256 x 256 to speed up segmentation of
    large images.
    :param image_path:
        Path to the image
    :param n_segments:
        number of segments per image
    :return segments:
        The segments of the image
    """
    test_image = imread(image_path)
    img_size_orig = test_image.shape[:2]
    test_image_resize = resize(test_image, (256, 256))
    segments = slic(test_image_resize, n_segments=n_segments, compactness=5)
    if len(np.unique(segments)) < 2:
        segments = slic(test_image_resize, n_segments=n_segments, compactness=100)

    segments_size_corrected = resize(segments, img_size_orig, preserve_range=True)

    return segments_size_corrected.astype(int)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--image_dir', type=str, default='train_data')
    parser.add_argument('--save_dir', type=str, default='masks')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # for filename in glob('/content/generative-liminal-space/train_data/*'):
    for filename in glob('scraped_data/*'):
        masks = segment_image(filename, args.N)
        file_basename = os.path.basename(filename).split('.')[0]
        for mask_index in np.unique(masks):
            mask = (masks != mask_index).astype(float)
            img = Image.fromarray(mask * 255).convert('1')
            img.save('{:s}/{:s}_{:06d}.jpg'.format(args.save_dir, file_basename, mask_index))
