import cv2
import torch
import numpy as np
import glob, random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from utils import imread_uint, rgb2ycbcr, augment_img, uint2tensor3, read_image, augment_imgs

suffixes = ['/*.png', '/*.jpg', '/*.bmp', '/*.tif']


# ---------------------- Gaussian Noise ---------------------- #
class train_syn_from_clean_gn(Dataset):
    def __init__(self, gray, sigma, patch_size, src_data):
        super().__init__()
        self.gray = gray
        self.sigma = sigma / 255.0
        self.data = []
        for suffix in suffixes:
            self.data.extend(glob.glob(src_data + suffix))
        self.data = sorted(self.data)
        self.transform = transforms.Compose([
            transforms.RandomCrop((patch_size, patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Lambda(
                lambda img: F.rotate(img, 90) if random.random() > 0.5 else img
            ),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        clean = Image.open(self.data[index])
        if self.gray:
            clean = clean.convert('L')
        else:
            clean = clean.convert('RGB')
        clean = self.transform(clean)
        noise = torch.randn(clean.size()) * self.sigma
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean

    def __len__(self):
        return len(self.data)


class test_syn_from_clean_gn(Dataset):
    def __init__(self, gray, sigma, src_data, test_seed=10):
        super().__init__()
        self.gray = gray
        self.seed = test_seed
        self.sigma = sigma / 255.0
        self.data = []
        for suffix in suffixes:
            self.data.extend(glob.glob(src_data + suffix))
        self.data = sorted(self.data)
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        torch.manual_seed(self.seed + 1)
        clean = Image.open(self.data[index])
        if self.gray:
            clean = clean.convert('L')
        else:
            clean = clean.convert('RGB')
        clean = self.transform(clean)
        H, W = clean.size()[-2:]
        clean = clean[..., :H // 16 * 16, :W // 16 * 16]
        noise = torch.randn(clean.size()) * self.sigma
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean

    def __len__(self):
        return len(self.data)


# ---------------------- JPEG Compression Artifact Reduction---------------------- #
class train_syn_from_clean_jpegcar(Dataset):
    def __init__(self, patch_size, quality_factor, src_data):
        super().__init__()
        self.patch_size = patch_size
        self.quality_factor = quality_factor
        self.data = []
        for suffix in suffixes:
            self.data.extend(glob.glob(src_data + suffix))
        self.data = sorted(self.data)

    def __getitem__(self, index):
        img_L = imread_uint(self.data[index])
        quality_factor = self.quality_factor
        if random.random() > 0.5:
            img_L = rgb2ycbcr(img_L)
        else:
            img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2GRAY)

        # randomly crop a large patch
        H, W = img_L.shape[:2]
        self.patch_size_plus = self.patch_size + 8

        # ---------------------------------
        # randomly crop a large patch
        # ---------------------------------
        rnd_h = random.randint(0, max(0, H - self.patch_size_plus))
        rnd_w = random.randint(0, max(0, W - self.patch_size_plus))
        patch_H = img_L[rnd_h:rnd_h + self.patch_size_plus, rnd_w:rnd_w + self.patch_size_plus, ...]

        # augment
        mode = random.randint(0, 7)
        img_L = augment_img(img_L, mode=mode)

        img_H = img_L.copy()
        result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img_L = cv2.imdecode(encimg, 0)

        # random crop
        H, W = img_H.shape[:2]
        if random.random() > 0.5:
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
        else:
            rnd_h = 0
            rnd_w = 0
        img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]

        img_L, img_H = uint2tensor3(img_L), uint2tensor3(img_H)
        return img_L, img_H

    def __len__(self):
        return len(self.data)


class test_syn_from_clean_jpegcar(Dataset):
    def __init__(self, quality_factor, src_data):
        super().__init__()
        self.quality_factor = quality_factor
        self.data = []
        for suffix in suffixes:
            self.data.extend(glob.glob(src_data + suffix))
        self.data = sorted(self.data)
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_H = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
        is_to_ycbcr = True if img_H.ndim == 3 else False
        if is_to_ycbcr:
            img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)
            img_H = rgb2ycbcr(img_H)

        quality_factor = self.quality_factor
        result, encimg = cv2.imencode('.jpg', img_H, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img_L = cv2.imdecode(encimg, 0)

        img_L, img_H = uint2tensor3(img_L), uint2tensor3(img_H)
        H, W = img_L.shape[-2:]
        img_L = img_L[..., :H // 16 * 16, :W // 16 * 16]
        img_H = img_H[..., :H // 16 * 16, :W // 16 * 16]

        return img_L, img_H

    def __len__(self):
        return len(self.data)


# ---------------------- Motion Deblur ---------------------- #
class train_image_pair_md(Dataset):
    def __init__(self, lr_root, gt_root, patch_size):
        super().__init__()
        self.patch_size = patch_size

        self.lr_data = []
        self.gt_data = []
        for suffix in suffixes:
            self.lr_data.extend(glob.glob(lr_root + suffix))
            self.gt_data.extend(glob.glob(gt_root + suffix))
        self.lr_data = sorted(self.lr_data)
        self.gt_data = sorted(self.gt_data)

        assert len(self.lr_data) == len(self.gt_data), "the length of lrs and gts is not equal!"

    def __getitem__(self, index):
        lr_img = read_image(self.lr_data[index])
        gt_img = read_image(self.gt_data[index])

        H, W, _ = lr_img.shape

        rnd_h = random.randint(0, H - self.patch_size)
        rnd_w = random.randint(0, W - self.patch_size)
        lr_img = lr_img[rnd_h: rnd_h + self.patch_size, rnd_w: rnd_w + self.patch_size]
        gt_img = gt_img[rnd_h: rnd_h + self.patch_size, rnd_w: rnd_w + self.patch_size]

        lr_img, gt_img = augment_imgs([lr_img, gt_img])

        lr_img = torch.from_numpy(np.ascontiguousarray(lr_img)).permute(2, 0, 1).float()
        gt_img = torch.from_numpy(np.ascontiguousarray(gt_img)).permute(2, 0, 1).float()

        return lr_img, gt_img

    def __len__(self):
        return len(self.lr_data)


class test_blur_image_md(Dataset):
    def __init__(self, lr_root, gt_root):
        super().__init__()

        self.lr_data = []
        self.gt_data = []
        for suffix in suffixes:
            self.lr_data.extend(glob.glob(lr_root + suffix))
            self.gt_data.extend(glob.glob(gt_root + suffix))
        self.lr_data = sorted(self.lr_data)
        self.gt_data = sorted(self.gt_data)

        assert len(self.lr_data) == len(self.gt_data), "the length of lrs and gts is not equal!"

    def __getitem__(self, index):
        lr_img = read_image(self.lr_data[index])
        gt_img = read_image(self.gt_data[index])

        H, W, _ = lr_img.shape
        lr_img = lr_img[:H // 16 * 16, :W // 16 * 16]
        gt_img = gt_img[:H // 16 * 16, :W // 16 * 16]

        lr_img = torch.from_numpy(np.ascontiguousarray(lr_img)).permute(2, 0, 1).float()
        gt_img = torch.from_numpy(np.ascontiguousarray(gt_img)).permute(2, 0, 1).float()
        return lr_img, gt_img

    def __len__(self):
        return len(self.lr_data)
