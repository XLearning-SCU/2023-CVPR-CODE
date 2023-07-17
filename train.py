import os
import torch
import numpy as np
import argparse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model import CODEFormer
from dataloader import train_image_pair_md, train_syn_from_clean_gn, train_syn_from_clean_jpegcar
from dataloader import test_blur_image_md, test_syn_from_clean_gn, test_syn_from_clean_jpegcar

from utils import create_dir, logger, set_random_seed, CharbonnierLoss

parser = argparse.ArgumentParser(description='CODEFormer')
parser.add_argument('--task', type=str, default='gdn', help='task to be performed: gdn, jpeg or md')
parser.add_argument('--sigma', type=int, default=15,
                    help='noise level in denoising or compression level in jpeg compression artifact reduction')
parser.add_argument('--in_chans', type=int, default=1, help='the input channel number to model')

parser.add_argument('--train_data_root', type=str, default='dataset/', help='data image path')
parser.add_argument('--val_data_root', type=str, default='dataset/', help='validation image path')
parser.add_argument('--exp_dir', type=str, default='experiments/', help='experiment dir')
parser.add_argument('--log_dir', type=str, default='results.txt', help='log file')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='the local rank of process')
parser.add_argument('--dist', action='store_true', help='activate distributed data parallel')

parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--patch_size', type=int, default=128, help='patch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--loss', type=str, default='L2', help='training loss')
parser.add_argument('--train_iter', type=int, default=800000, help='train iterations')
parser.add_argument('--print_iter', type=int, default=200, help='print interval')
parser.add_argument('--val_iter', type=int, default=5000, help='validation interval')
parser.add_argument('--save_iter', type=int, default=5000, help='save model every n iterations')
args = parser.parse_args()


def train():
    if args.local_rank == 0:
        create_dir(args.exp_dir)
        logging = logger(args.log_dir)

    if args.local_rank == 0:
        logging.info('Available GPUs: {}'.format(torch.cuda.device_count()))

    set_random_seed(args.seed)
    if args.local_rank == 0:
        logging.info('Random seed: {}'.format(args.seed))

    if args.dist:
        dist = True
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        world_size = torch.distributed.get_world_size()
        if args.local_rank <= 0:
            logging.info(f'World Size: {world_size}')
    else:
        dist = False

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if args.task == 'gdn' or args.task == 'jpeg':
        args.in_chans = 1

    model = CODEFormer(args.in_chans)

    if torch.cuda.is_available():
        model = model.cuda()
        if args.dist:
            model = DistributedDataParallel(model, device_ids=[args.local_rank])
            if args.local_rank == 0:
                logging.info('Enable DistributedDataParallel for training')
        else:
            if args.local_rank == 0:
                logging.info('Enable one GPU for training')
    else:
        if args.local_rank == 0:
            logging.info('Enable CPU for training')

    if args.dist and args.local_rank >= 0:
        world_size = torch.distributed.get_world_size()

    if args.task == 'md':
        train_set = train_image_pair_md(os.path.join(args.train_data_root, 'input'),
                                        os.path.join(args.train_data_root, 'target'), args.patch_size)
        val_set = test_blur_image_md(os.path.join(args.val_data_root, 'input'),
                                     os.path.join(args.val_data_root, 'target'))
    elif args.task == 'jpeg':
        train_set = train_syn_from_clean_jpegcar(args.patch_size, args.sigma, args.train_data_root)
        val_set = test_syn_from_clean_jpegcar(args.sigma, args.val_data_root)
    elif args.task == 'gdn' or args.task == 'cdn':
        train_set = train_syn_from_clean_gn(args.task == 'gdn', args.sigma, args.patch_size, args.train_data_root)
        val_set = test_syn_from_clean_gn(args.task == 'gdn', args.sigma, args.val_data_root)
    else:
        raise NotImplementedError("Undefined Recognized Task! Define the dataset in dataloader and add it here.")

    dist_sampler = DistributedSampler(train_set) if args.dist else None
    train_data = DataLoader(
        train_set, batch_size=args.batch_size // world_size if args.dist else args.batch_size,
        shuffle=False if args.dist else True, sampler=dist_sampler, num_workers=4, drop_last=True)
    valid_data = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    if args.local_rank == 0:
        logging.info('Number of training images: {:d}'.format(len(train_set)))
        logging.info('Number of validation images: {:d}'.format(len(valid_data)))
        logging.info('Number of training iterations: {:d}'.format(args.train_iter))

    if args.loss == 'L2':
        criterion = torch.nn.MSELoss()
        if args.local_rank == 0:
            logging.info('Enable L2 loss for training')
    elif args.loss == 'L1':
        criterion = torch.nn.L1Loss()
        if args.local_rank == 0:
            logging.info('Enable L1 loss for training')
    elif args.loss == 'Char':
        criterion = CharbonnierLoss(eps=1e-3)
        if args.local_rank == 0:
            logging.info('Enable Chabonnier loss for training')
    else:
        raise NotImplementedError("Undefined Loss!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.train_iter, eta_min=1e-6)

    if args.local_rank == 0:
        logging.info('Begin Training......')
    curr_itrs = 0
    while True:
        if curr_itrs > args.train_iter:
            break
        if args.dist:
            dist_sampler.set_epoch(curr_itrs // len(train_data))

        for inputs, labels in train_data:
            curr_itrs += 1
            if curr_itrs > args.train_iter:
                break

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs.detach())
            loss = criterion(outputs, labels.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.dist: torch.distributed.all_reduce(
                loss.clone().detach(), op=torch.distributed.ReduceOp.SUM
            )
            if args.local_rank == 0 and curr_itrs % args.print_iter == 0:
                logging.info("Iteration: {:0>3}, Loss: {:.8f}".format(curr_itrs, loss.item()))

            if args.local_rank == 0 and curr_itrs % args.val_iter == 0:
                net = model.module if args.dist else model
                net.eval()
                psnr, ssim = 0, 0
                for inputs, labels in valid_data:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()

                    with torch.no_grad():
                        outputs = net(inputs)

                    outputs = outputs[0].cpu().numpy().transpose((1, 2, 0)).squeeze().clip(0, 1)
                    labels = labels[0].numpy().transpose((1, 2, 0)).squeeze()

                    psnr += compare_psnr(labels, outputs, data_range=1.0)
                    ssim += compare_ssim(labels, outputs, data_range=1.0,
                                         multichannel=False if len(outputs.shape) == 2 else True)

                logging.info('PSNR: {:.3f}, SSIM:{:.5f}'.format(
                    psnr / len(valid_data), ssim / len(valid_data)
                ))
                net.train()

            if args.local_rank == 0 and curr_itrs % args.save_iter == 0:
                state_dict = model.module.state_dict() \
                    if args.dist else model.state_dict()
                torch.save(state_dict, os.path.join(args.exp_dir, 'model_%04d_dict.pth' % (curr_itrs)))
            if dist: torch.distributed.barrier()

    if args.local_rank == 0: logging.info('End of the training.')


if __name__ == '__main__':
    train()
