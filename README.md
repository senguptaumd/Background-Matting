# Background Matting: The World is Your Green Screen
![alt text](https://homes.cs.washington.edu/~soumya91/paper_thumbnails/matting.png)

By Soumyadip Sengupta, Vivek Jayaram, Brian Curless, Steve Seitz, and Ira Kemelmacher-Shlizerman

This paper will be presented in IEEE CVPR 2020.

## [**Project Page**](http://grail.cs.washington.edu/projects/background-matting/)

Go to Project page for additional details and results.

## [Paper (Arxiv)](https://arxiv.org/abs/2004.00626)

## Project members ##

* [Soumyadip Sengupta](https://homes.cs.washington.edu/~soumya91/), University of Washington
* [Vivek Jayaram](http://www.vivekjayaram.com/research), University of Washington
* [Brian Curless](https://homes.cs.washington.edu/~curless/), University of Washington
* [Steve Seitz](https://homes.cs.washington.edu/~seitz/), University of Washington
* [Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/), University of Washington

### License ###
This work is licensed under the [Creative Commons Attribution NonCommercial ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Summary ##
- [Updates](#updates) 
- [Getting Started](#getting-started)
- [Inference Code on images](#run-the-inference-code-on-sample-images)
- [Inference Code on videos](#run-the-inference-code-on-sample-videos)
- [Notes on capturing images](#notes-on-capturing-images)
- [Training code (coming soon ...)](#training-code)
- [Captured Data (coming soon ...)](#dataset)
- [Citations](#citation)

## **Updates** ##
April 8, 2020
- [Testing code to replace background for videos](#run-the-inference-code-on-sample-videos)
- Bug fixes
	- Turning off `adjustExposure()` for bias-gain correction in `test_pre_processing.py`. (Bug found, need to be fixed)
	- Incorporating 'uncropping' operation in `test_background-matting_image.py`. (Output will be of same resolution and aspect-ratio as input)


## Getting Started 

Clone repository: 
```
git clone https://github.com/senguptaumd/Background-Matting.git
```

Please use Python 3. Create an [Anaconda](https://www.anaconda.com/distribution/) environment and install the dependencies. Our code is tested with Pytorch=1.1.0, Tensorflow=1.14 with cuda10.0

```
conda create --name back-matting python=3.6
conda activate back-matting
```
Make sure CUDA 10.0 is your default cuda. If your CUDA 10.0 is installed in `/usr/local/cuda-10.0`, apply
```
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
``` 
Install PyTorch, Tensorflow (needed for segmentation) and dependencies
```
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
pip install tensorflow-gpu=1.14.0
pip install -r requirements.txt

```

Note: The code is likely to work on other PyTorch and Tensorflow versions compatible with your system CUDA. If you already have a working environment with PyTorch and Tensorflow, only install dependencies with `pip install -r requirements.txt`. If our code fails due to different versions, then you need to install specific CUDA, PyTorch and Tensorflow versions.

## Run the inference code on sample images

### Data

To perform Background Matting based green-screening, you need to capture:
- (a) Image with the subject (use `_img.png` extension)
- (b) Image of the background without the subject (use `_back.png` extension)
- (c) Target background to insert the subject (place in `data/background`)

Use `sample_data/` folder for testing and prepare your own data based on that.

### Pre-trained model

Please download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1WLDBC_Q-cA72QC8bB-Rdj53UB2vSPnXv?usp=sharing) and place `Models/` folder inside `Background-Matting/`.


### Pre-processing

1. Segmentation

Background Matting needs a segmentation mask for the subject. We use tensorflow version of [Deeplabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab).

```
cd Background-Matting/
git clone https://github.com/tensorflow/models.git
cd models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../..
python test_segmentation_deeplab.py -i sample_data/input
```

You can replace Deeplabv3+ with any segmentation network of your choice. Save the segmentation results with extension `_masksDL.png`.

2. Alignment

- For hand-held camera, we need to align the background with the input image as a part of pre-processing. We apply simple hoomography based alignment.
- We ask users to **disable the auto-focus and auto-exposure** of the camera while capturing the pair of images. This can be easily done in iPhone cameras (tap and hold for a while).

Run `python test_pre_process.py -i sample_data/input` for pre-processing. It aligns the background image `_back.png` and changes its bias-gain to match the input image `_img.png`

### Background Matting

```bash
python test_background-matting_image.py -m real-hand-held -i sample_data/input/ -o sample_data/output/ -tb sample_data/background/0001.png
```
For images taken with fixed camera (with a tripod), choose `-m real-fixed-cam` for best results. `-m syn-comp-adobe` lets you use the model trained on synthetic-composite Adobe dataset, without real data (worse performance).

## Run the inference code on sample videos

This is almost exactly similar as that of the image with few small changes.

### Data

To perform Background Matting based green-screening, you need to capture:
- (a) Video with the subject (`teaser.mov`)
- (b) Image of the background without the subject (use `teaser_back.png` extension)
- (c) Target background to insert the subject (place in `fountain.mov`)

Use `sample_video/` folder for testing and prepare your own data based on that.

### Pre-processing

1. Frame extraction:

```
cd Background-Matting/sample_video
ffmpeg -i teaser.mov input/%04d_img.png -hide_banner
ffmpeg -i fountain.mov background/%04d.png -hide_banner
```

2. Segmentation
```
cd Background-Matting/
git clone https://github.com/tensorflow/models.git
cd models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../..
python test_segmentation_deeplab.py -i sample_video/input
```

3. Alignment
Run `python test_pre_process_video.py -i sample_video/input -v_name sample_video/teaser_back.png` for pre-processing.

### Background Matting

```bash
python test_background-matting_image.py -m real-hand-held -i sample_video/input/ -o sample_video/output/ -tb sample_video/background/
```

To obtain the video from the output frames, run:
```
cd Background-Matting/sample_video
ffmpeg -r 60 -f image2 -i output/%04d_matte.png -vcodec libx264 -crf 15 -s 1280x720 -pix_fmt yuv420p teaser_matte.mp4
ffmpeg -r 60 -f image2 -i output/%04d_compose.png -vcodec libx264 -crf 15 -s 1280x720 -pix_fmt yuv420p teaser_compose.mp4
```

## Notes on capturing images

For best results capture images following these guidelines:
- Choose a background that is mostly static, can be both indoor and outdoor.
- Avoid casting any shadows of the subject on the background.
	- place the subject atleast few feets away from the background.
	- if possible adjust the lighting to avoid strong shadows on the background.
- Avoid large color coincidences between subject and background. (e.g. Do not wear a white shirt in front of a white wall background.)
- Lock AE/AF (Auto-exposure and Auto-focus) of the camera.
- For hand-held capture, you need to:
	- allow only small camera motion by continuing to holding the camera as the subject exists the scene.
	- avoid backgrounds that has two perpendicular planes (homography based alignment will fail) or use a background very far away.
	- The above restirctions do not apply for images captured with fixed camera (on a tripod)

	 
## Training code

We will also release the training code, which will allow users to train on labelled data and also on unlabelled real data.

**Coming soon ...**

## Dataset

We collected 50 videos with both fixed and hand-held camera in indoor and outdoor settings. We plan to release this data to encourage future research on improving background matting.

**Coming soon ...**

## Notes

We are eager to hear how our algorithm works on your images/videos. If the algorithm fails on your data, please feel free to share it with us at soumya91@cs.washington.edu. This will help us in improving our algorithm for future research. Also, feel free to share any cool results.

## Citation
If you use this code for your research, please consider citing:
```
@InProceedings{BMSengupta20,
  title={Background Matting: The World is Your Green Screen},
  author = {Soumyadip Sengupta and Vivek Jayaram and Brian Curless and Steve Seitz and Ira Kemelmacher-Shlizerman},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2020}
}
```
