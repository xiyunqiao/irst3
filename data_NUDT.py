import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps
import os.path as osp
import random
import numpy as np
import cv2


class MyDataset(Data.Dataset):
    def __init__(self, mode):
        # path of dataset
        path = './dataset/'
        dataset = 'NUDT-SIRST'
        mode = mode
        base_dir = path + dataset + '/'

        mean = 0.3428
        std = 0.2315
        self.crop_size = 256
        self.base_size = 256
        # list of images in dataset
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.mask_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([mean], [std])
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = self.imgs_dir + '/' + name + '.png'
        mask_path = self.mask_dir + '/' + name + '.png'

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('1')
        w, h = img.size[0], img.size[1]

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask

def sobel_with_gf(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # 进行引导滤波（假设guide和src是相同的）
    filtered_img = cv2.ximgproc.guidedFilter(guide=img, src=img, radius=4, eps=0.04)

    # # 2倍下采样
    downsampled_img = cv2.resize(filtered_img, (filtered_img.shape[1] // 2, filtered_img.shape[0] // 2),
                                 interpolation=cv2.INTER_LINEAR)

    # 2倍上采样
    upsampled_img = cv2.resize(downsampled_img, (downsampled_img.shape[1] * 2, downsampled_img.shape[0] * 2),
                               interpolation=cv2.INTER_LINEAR)

    # Sobel边缘提取
    # 转为灰度图
    gray_img = cv2.cvtColor(upsampled_img, cv2.COLOR_BGR2GRAY)
    return gray_img

class MyEdgeSet(Data.Dataset):
    def __init__(self, mode):
        # path of dataset
        path = './dataset/'
        dataset = 'NUDT-SIRST'
        mode = mode
        base_dir = path + dataset + '/'

        mean = 0.3428
        std = 0.2315
        self.crop_size = 256
        self.base_size = 256
        # list of images in dataset
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'val':
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.img_sobel_dir = osp.join(base_dir, 'images_guidedfilter')
        self.mask_dir = osp.join(base_dir,'masks')
        self.mask_edge_dir = osp.join(base_dir, 'masks_edge')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([mean], [std])
        ])


    def __getitem__(self, i):
        name = self.names[i]
        img_path = self.imgs_dir + '/' + name + '.png'
        mask_path = self.mask_dir + '/' + name + '.png'

        img = Image.open(img_path).convert('L')
        img_sobel = sobel_with_gf(img_path)
        mask = Image.open(mask_path).convert('1')

        w, h = img.size[0], img.size[1]


        img, mask = self.transform(img), transforms.ToTensor()(mask)
        img_sobel = self.transform(img_sobel)
        return img, img_sobel, mask

    def __len__(self):
        return len(self.names)
