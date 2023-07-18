PyTorch implementation for Comprehensive and Delicate: An Efficient Transformer for Image Restoration (CVPR 2023).

## Dependencies

* Python 3.7.0
* PyTorch 1.11.0+cu113

## Dataset

You could find the dataset we used in the paper at following:

### Dataset for Training

Denoising & JPEG compression artifact reduction: BSD400, DIV2K, Flickr2K, WaterlooED

Motion deblur: GoPro

### Dataset for Testing

Grayscale denoising: Set12, BSD68

Color denoising: CBSD68, Kodak24, McMaster

JPEG compression artifact reduction: Classic5, LIVE1

Motion deblur: GoPro test, HIDE

## Checkpoints

The checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1z32Wuphcq28WjwTEP1b1iPcVt7tJI1Iy?usp=share_link) Here.

They can be placed at the `ckpt` folder.

## Testing

~~~shell
python test.py --task jpeg --data_root dataset --dataset classic5 --sigma 40 --ckpt_pth ./ckpt/jpegcar_q40.pth --result_dir results/jpeg_q40
~~~


## Training

If you want to re-train our model, you need to first put the training set and validation dataset into `dataset` 
and use the command below:

~~~shell
python train.py --task YourTask --sigma NoiseParameter --train_data_root TraindataRoot --loss L2 --train_iter 500000 --val_iter
~~~

## Citation

If you find our work useful in your research, please consider citing:

~~~
@inproceedings{zhao2023comprehensive,
  title={Comprehensive and Delicate: An Efficient Transformer for Image Restoration},
  author={Zhao, Haiyu and Gou, Yuanbiao and Li, Boyun and Peng, Dezhong and Lv, Jiancheng and Peng, Xi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14122--14132},
  year={2023}
}
~~~

## Acknowledgement

