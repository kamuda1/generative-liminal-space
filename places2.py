import os
import random
import torch
import numpy as np
from PIL import Image
from glob import glob

from generate_data import segment_image


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('/content/generative-liminal-space/train_data/*')
        else:
            self.paths = glob('/content/generative-liminal-space/train_data/*')

        self.mask_paths = glob('/content/masks/*')
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        filename = self.paths[index]
        gt_img = Image.open(filename)
        gt_img = self.img_transform(gt_img.convert('RGB'))

        file_basename = os.path.basename(filename).split('.')[0]
        possible_masks = glob(os.path.join('/content/masks/',
                                           file_basename + '*'))
        if len(possible_masks) == 0:
            masks = segment_image(filename, 5)
            for mask_index in np.unique(masks):
                tmp_mask = (masks != mask_index).astype(float)
                tmp_mask = Image.fromarray(tmp_mask * 255).convert('1')
                possible_masks.append(tmp_mask)
            mask = random.choice(possible_masks)
        else:
          mask = Image.open(random.choice(possible_masks))
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)