## Training on synthetic-composite Adobe dataset ##

### Data ###

- Download original Adobe matting dataset: [Follow instructions.](https://sites.google.com/view/deepimagematting)
- Separate human images: Use `test_data_list.txt` and `train_data_list.txt` in `Data_adobe` to copy only human subjects from Adobe dataset. Create folders `fg_train`, `fg_test`, `mask_train`, `mask_test` to copy foreground and alpha matte for test and train data separately. (The train test split is same as the original dataset.) You can run the following to accomplish this:
```bash
cd Data_adobe
./prepare.sh /path/to/adobe/Combined_Dataset
```
- Download background images: Download MS-COCO images and place it in [`bg_train`](http://images.cocodataset.org/zips/test2017.zip) and in [`bg_test`](http://images.cocodataset.org/zips/val2017.zip).
- Compose Adobe foregrounds onto COCO for the train and test sets. This saves the composed result as `_comp` and the background as `_back` under `merged_train` and `merged_test`. It will also create a CSV to be used by the training dataloader. You can pass `--workers 8` to use e.g. 8 threads, though it will use only one by default.
```bash
python compose.py --fg_path fg_train --mask_path mask_train --bg_path bg_train --out_path merged_train --out_csv Adobe_train_data.csv
python compose.py --fg_path fg_test --mask_path mask_test --bg_path bg_test --out_path merged_test
```


### Training ###

Change number of GPU and required batch-size, depending on your platform. We trained the model with 512x512 input (`-res` flag).

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_adobe.py -n Adobe_train -bs 4 -res 512
```

Notes:
- 512x512 is the maximum input resolution we recommend for training
- If you decreasing training resolution to 256x256, change `-res 256`, but we also recommend using lesser residual blocks. Use: `-n_blocks1 5 -n_blocks2 2`.

Cheers to the [unofficial Deep Image Matting repo.](https://github.com/foamliu/Deep-Image-Matting-PyTorch)

## Training on unlabeled real videos ##

### Data ###

[Please download our captured videos.](https://drive.google.com/drive/folders/1j3BMrRFhFpfzJAe6P2WDtfanoeSCLPiq?usp=sharing). We will show next how to finetune your model on `fixed-camera` captured videos. It will be similar for `hand-held` cameras, except you will need to align the captured background image to each frame of the video separately. (Take a hint from `test_pre_process.py` and use `alignImages()`.) 

Data Pre-processing: 
	- Extract frames for each video: `ffmpeg -i $NAME.mp4 $NAME/%04d_img.png -hide_banner`
	- Run Segmentation (follow instructions on Deeplabv3+) : `python test_segmentation_deeplab.py -i $NAME`
	- Target background for composition. For self-supervised learning we need some target backgrounds that has roughly similar lighting as the original videos. Either capture few videos of indoor/outdoor scenes without humans or use our captured background in the `background` folder.
	- Create a .csv file `Video_data_train.csv` with each row as: `$image;$captured_back;$segmentation;$image+20frames;$image+2*20frames;$image+3*20frames;$image+4*20frames;$target_back`[TODO Andrey].

### Training ###

Change number of GPU and required batch-size, depending on your platform. We trained the model with 512x512 input (`-res` flag).

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_real_fixed.py -n Real_fixed -bs 4 -res 512 -init_model Models/syn-comp-adobe-trainset/net_epoch_64.pth
```

