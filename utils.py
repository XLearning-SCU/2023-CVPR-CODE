import cv2
import torch
import os, sys
import numpy as np
import random, logging
import torch.nn.functional as F


# --------------------- Data Utils ------------------------- #
def augment_img(img, mode=0):
    """Kai Zhang (github: https://github.com/cszn)
    """
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def augment_imgs(img_list, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    return img


def rgb2ycbcr(img, only_y=True):
    """
    same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


# --------------------- Train/Test Utils ------------------------- #
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def logger(log_file=None):
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p'
    )
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logging.getLogger().addHandler(fh)
    return logging


# --------------------- Ensembling Utils ------------------------- #
def image2patch(image, patch_size, step):
    '''convert image to patches'''
    assert len(image.size()) == 4 and image.size(0) == 1
    H, W = image.size()[-2:]
    h_space = list(range(0, H - patch_size, step))
    h_space.append(H - patch_size)
    w_space = list(range(0, W - patch_size, step))
    w_space.append(W - patch_size)
    patches = [
        image[..., h:h + patch_size, w:w + patch_size] \
        for h in h_space \
        for w in w_space
    ]
    patches = torch.cat(patches, dim=0)
    return patches


def patch2image(patches, height, width, step):
    image = patch2image_wo_norm(patches, height, width, step)
    norm = patch2image_wo_norm(image2patch(
        torch.ones_like(image), patches.size(-1), step
    ), height, width, step)
    return image / norm


def patch2image_wo_norm(patches, height, width, step):
    '''convert patches to image'''
    assert len(patches.size()) == 4 and patches.size(-2) == patches.size(-1)
    _, C, patch_size, _ = patches.size()
    image = torch.zeros(1, C, height, width, device=patches.device)
    h_space = list(range(0, height - patch_size, step))
    h_space.append(height - patch_size)
    w_space = list(range(0, width - patch_size, step))
    w_space.append(width - patch_size)
    idx = 0
    for h in h_space:
        for w in w_space:
            image[..., h:h + patch_size, w:w + patch_size] += patches[idx]
            idx += 1
    return image


def augment_imglist(img):
    def _transform(v, op):
        v2np = v.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()
        return ret

    x = [img]
    for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])
    return x


def reverse_augment(img_list):
    def _transform(v, op):
        v2np = v.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()
        return ret

    B, C, H, W = img_list[0].shape
    for i in range(len(img_list)):
        if i > 3:
            img_list[i] = _transform(img_list[i], 't')
        if i % 4 > 1:
            img_list[i] = _transform(img_list[i], 'h')
        if (i % 4) % 2 == 1:
            img_list[i] = _transform(img_list[i], 'v')

    img_list = [img.view(1, B, C, H, W) for img in img_list]
    y = [torch.cat(img_list, dim=0).mean(dim=0, keepdim=False)]
    y = y[0]
    return y


def model_x8(inputs, model):
    inputs_list = augment_imglist(inputs)
    with torch.no_grad():
        outputs = [model(input) for input in inputs_list]
    ensembled_outputs = reverse_augment(outputs)
    outputs = ensembled_outputs
    return outputs


# --------------------- Metric Utils ------------------------- #
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).
    It is called by func:`calculate_ssim_pt`.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def calculate_psnr_pt(img, img2, crop_border=0, test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).
    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


def calculate_ssim_pt(img, img2, crop_border=0, test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity) (PyTorch version).
    ``Paper: Image quality assessment: From error visibility to structural similarity``
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


# --------------------- Training Utils ------------------------- #
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CharbonnierLoss(torch.nn.Module):
    """CharbonnierLoss."""
    def __init__(self, eps):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        error = torch.sqrt((pred - target)**2 + self.eps)
        loss = torch.mean(error)
        return loss