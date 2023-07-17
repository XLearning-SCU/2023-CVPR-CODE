import os
import torch
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from utils import create_dir, logger
from dataloader import test_syn_from_clean_gn, test_syn_from_clean_jpegcar, test_blur_image_md
from utils import image2patch, patch2image, model_x8
# from utils import calculate_psnr_pt, calculate_ssim_pt

from model import CODEFormer

parser = argparse.ArgumentParser(description='CODEFormer')
parser.add_argument('--task', type=str, default='gdn', help='task to be performed: gdn, cdn, jpeg or md')
parser.add_argument('--sigma', type=int, default=15,
                    help='noise level in denoising or compression level in jpeg compression artifact reduction')
parser.add_argument('--data_root', type=str, default='dataset', help='data image path')
parser.add_argument('--dataset', type=str, default='Set12', help='testing datasets')
parser.add_argument('--ensemble', action='store_true', help='if ensembling or not')
parser.add_argument('--save_img', action='store_true', help='save result images')
parser.add_argument('--ckpt_pth', type=str, default='./ckpt/gdn_sig15.pth', help='model path')
parser.add_argument('--result_dir', type=str, default='results/gdn_sig15', help='results dir')
parser.add_argument('--log_dir', type=str, default='log.txt', help='log file')
args = parser.parse_args()


def test():
    create_dir(args.result_dir)
    logging = logger(os.path.join(args.result_dir, args.log_dir))

    if args.task == 'gdn' or args.task == 'jpeg':
        in_chans = 1
    else:
        in_chans = 3

    model = CODEFormer(in_chans)
    model.load_state_dict(torch.load(args.ckpt_pth))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    if args.task == 'md':
        test_data = DataLoader(test_blur_image_md(os.path.join(args.data_root, args.dataset, 'input'),
                                                  os.path.join(args.data_root, args.dataset, 'target')),
                               batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    elif args.task == 'jpeg':
        test_data = DataLoader(test_syn_from_clean_jpegcar(args.sigma, os.path.join(args.data_root, args.dataset)),
                               batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    elif args.task == 'gdn' or args.task == 'cdn':
        test_data = DataLoader(test_syn_from_clean_gn(args.task == 'gdn', args.sigma,
                                                      os.path.join(args.data_root, args.dataset)),
                               batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    else:
        raise NotImplementedError("Undefined Task!!!")

    total_psnr, total_ssim = 0, 0
    for idx, (inputs, labels) in enumerate(test_data):
        if args.task == 'md':
            data_name = os.path.basename(test_data.dataset.lr_data[idx])
        else:
            data_name = os.path.basename(test_data.dataset.data[idx])
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        if args.task != 'md':
            if args.ensemble:
                with torch.no_grad():
                    outputs = model_x8(inputs, model=model)
            else:
                with torch.no_grad():
                    outputs = model(inputs.detach())
        else:
            if args.ensemble:
                raise NotImplementedError("The self-ensemble strategy is not support for motion deblur.")
            lr_h, lr_w = inputs.size()[2:]
            lr_list = image2patch(inputs, 256, 250)
            with torch.no_grad():
                res = torch.cat([
                    model(lr) for lr in torch.split(lr_list, 256, dim=0)
                ], dim=0)
            outputs = patch2image(res, lr_h, lr_w, 250)

        outputs = outputs[0].cpu().numpy().transpose((1, 2, 0)).squeeze().clip(0, 1)
        labels = labels[0].numpy().transpose((1, 2, 0)).squeeze()

        psnr = compare_psnr(labels, outputs, data_range=1.0)
        ssim = compare_ssim(labels, outputs, data_range=1.0,
                            multichannel=False if len(outputs.shape) == 2 else True)
        total_psnr += psnr
        total_ssim += ssim

        logging.info('{} - PSNR: {:.3f} - SSIM:{:.5f}'.format(data_name, psnr, ssim))
        data_name = data_name.split('.')[-2] + '.png'
        if args.save_img:
            Image.fromarray(np.uint8(outputs * 255)).save(os.path.join(args.result_dir, data_name))

    logging.info('DATASET: {}, AVG_PSNR: {:.3f}, AVG_SSIM:{:.4f}'.format(args.dataset,
                                                                         total_psnr / len(test_data),
                                                                         total_ssim / len(test_data)))


if __name__ == '__main__':
    test()
