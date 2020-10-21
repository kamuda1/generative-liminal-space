import os
import random
import torch
from PIL import Image
from glob import glob


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
        possible_masks = glob(os.path.join(self.mask_paths, file_basename, '*'))
        mask = random.choice(possible_masks)
        # mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
