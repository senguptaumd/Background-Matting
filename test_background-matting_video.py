from __future__ import print_function

import argparse
import glob
import os
import sys

import torch.backends.cudnn as cudnn
import torch.nn as nn
import tqdm
from skimage.measure import label
from torch.autograd import Variable

from functions import *
from networks import ResnetConditionHR

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='Background Matting.')
parser.add_argument('-m', '--trained_model', type=str, default='real-fixed-cam',
                    choices=['real-fixed-cam', 'real-hand-held', 'syn-comp-adobe'],
                    help='Trained background matting model')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='Directory to save the output results. (required)')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory to load input images. (required)')
parser.add_argument('-tb', '--target_back', type=str,
                    help='Target background to put foreground on. Either path to an image or an directory.')
parser.add_argument('-b', '--back', type=str, default=None,
                    help='Captured background image for fixed-cam mode. In case of hand-held mode, leave empty')

args = parser.parse_args()
# input data path
data_path = args.input_dir

if os.path.isdir(args.target_back):
    args.video = True
    print('Using video mode')
else:
    args.video = False
    print('Using image mode')
    # target background path
    target_img = cv2.imread(args.target_back)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    # Green-screen background
    target_green_img = np.zeros(target_img.shape, dtype=np.uint8)
    target_green_img[..., 0] = 0
    target_green_img[..., 1] = 255
    target_green_img[..., 2] = 0

# initialize network
model_main_dir = 'Models/' + args.trained_model + '/'
model_filepath = glob.glob(model_main_dir + 'netG_epoch_*')[0]
print("Loading model", model_filepath)
net = ResnetConditionHR(input_nc=(3, 3, 1, 4), output_nc=4, n_blocks1=7, n_blocks2=3)
net = nn.DataParallel(net)
net.load_state_dict(torch.load(model_filepath))
net.cuda()
net.eval()
cudnn.benchmark = True
reso = (512, 512)  # input resolution to the network

# load captured background for video mode, fixed camera
if args.back is not None:
    back_img = cv2.imread(args.back)
    back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)

# output directory
result_path = args.output_dir

if not os.path.exists(result_path):
    os.makedirs(result_path)

video = cv2.VideoCapture(data_path)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

masks_video = cv2.VideoCapture(data_path.replace("_raw.mp4", "_masksDL.avi"))

if not "raw" in data_path:
    print("Wrong input file specified!")
    sys.exit(0)

# Create a video writer for cutout object placed on target
output_path = data_path.replace("raw", "out")
video_writer = cv2.VideoWriter(output_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (width, height))
print("Writing video to", output_path)

# Create video writer for matte. Use lossless png compression
output_path = data_path.replace("raw", "masks").replace("mp4", "avi")
masks_video_writer = cv2.VideoWriter(output_path,
                                     cv2.VideoWriter_fourcc(*'png '),
                                     fps,
                                     (width, height))
for i in tqdm.trange(num_frames):

    # original image
    ret, bgr_img = video.read()
    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    output_height = bgr_img.shape[0]
    output_width = bgr_img.shape[1]

    # segmentation mask
    ret, seg_mask = masks_video.read()
    seg_mask = seg_mask[:, :, 0]

    ## create the multi-frame
    multi_fr_w = np.zeros((output_height, output_width, 4))
    multi_fr_w[..., 0] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    multi_fr_w[..., 1] = multi_fr_w[..., 0]
    multi_fr_w[..., 2] = multi_fr_w[..., 0]
    multi_fr_w[..., 3] = multi_fr_w[..., 0]

    # Crop all images by the bbox of the rough segmentation mask
    try:
        bbox = get_bbox(seg_mask, R=output_height, C=output_width)
    except: # Catch error occurring if no object is currently visible
        masks_video_writer.write(np.zeros((output_height, output_width, 3), dtype=np.uint8))
        target_green_img = cv2.resize(target_green_img, (output_width, output_height))
        video_writer.write(cv2.cvtColor(target_green_img, cv2.COLOR_BGR2RGB))
        print("Skipping frame", i)
        continue

    try:
        crop_list = [bgr_img, back_img, seg_mask, multi_fr_w]
        crop_list = crop_images(crop_list, reso, bbox)
        bgr_img = crop_list[0]
        bg_im = crop_list[1]
        seg_mask = crop_list[2]
        multi_fr = crop_list[3]
    except: # Catch error occurring if no (big enough) object is currently visible
        masks_video_writer.write(np.zeros((output_height, output_width, 3), dtype=np.uint8))
        target_green_img = cv2.resize(target_green_img, (output_width, output_height))
        video_writer.write(cv2.cvtColor(target_green_img, cv2.COLOR_BGR2RGB))
        print("Error cropping", i)
        continue

    # Preprocess the rough segmentation mask
    kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seg_mask = seg_mask.astype(np.float32) / 255
    seg_mask[seg_mask > 0.2] = 1
    K = 25

    zero_id = np.nonzero(np.sum(seg_mask, axis=1) == 0)
    del_id = zero_id[0][zero_id[0] > 250]
    if len(del_id) > 0:
        del_id = [del_id[0] - 2, del_id[0] - 1, *del_id]
        seg_mask = np.delete(seg_mask, del_id, 0)
    seg_mask = cv2.copyMakeBorder(seg_mask, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)

    seg_mask = cv2.erode(seg_mask, kernel_er, iterations=10)
    seg_mask = cv2.dilate(seg_mask, kernel_dil, iterations=5)
    seg_mask = cv2.GaussianBlur(seg_mask.astype(np.float32), (31, 31), 0)
    seg_mask = (255 * seg_mask).astype(np.uint8)
    seg_mask = np.delete(seg_mask, range(reso[0], reso[0] + K), 0)

    # Convert images to torch and normalize to range [-1, 1]
    img = torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0)
    img = 2 * img.float().div(255) - 1
    bg = torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0)
    bg = 2 * bg.float().div(255) - 1
    rcnn_al = torch.from_numpy(seg_mask).unsqueeze(0).unsqueeze(0)
    rcnn_al = 2 * rcnn_al.float().div(255) - 1
    multi_fr = torch.from_numpy(multi_fr.transpose((2, 0, 1))).unsqueeze(0)
    multi_fr = 2 * multi_fr.float().div(255) - 1

    with torch.no_grad():
        img, bg, rcnn_al, multi_fr = Variable(img.cuda()), Variable(bg.cuda()), Variable(rcnn_al.cuda()), Variable(
            multi_fr.cuda())
        input_im = torch.cat([img, bg, rcnn_al, multi_fr], dim=1)

        alpha_pred, fg_pred_tmp = net(img, bg, rcnn_al, multi_fr)

        # for regions with alpha>0.95, simply use the image as fg
        al_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)
        fg_pred = img * al_mask + fg_pred_tmp * (1 - al_mask)

        alpha_out = to_image(alpha_pred[0, ...])

        # Filter alpha image by largest connected area
        labels = label((alpha_out > 0.05).astype(int))
        try:
            assert (labels.max() != 0)
        except:
            masks_video_writer.write(np.zeros((output_height, output_width, 3), dtype=np.uint8))
            target_green_img = cv2.resize(target_green_img, (output_width, output_height))
            video_writer.write(cv2.cvtColor(target_green_img, cv2.COLOR_BGR2RGB))
            print("Skipping frame", i)
            continue
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        alpha_out = alpha_out * largestCC
        alpha_out = (255 * alpha_out[..., 0]).astype(np.uint8)

        fg_out = to_image(fg_pred[0, ...])
        fg_out = fg_out * np.expand_dims((alpha_out.astype(float) / 255 > 0.01).astype(float), axis=2)
        fg_out = (255 * fg_out).astype(np.uint8)

        # Uncrop
        alpha_out = uncrop(alpha_out, bbox, output_height, output_width)
        fg_out = uncrop(fg_out, bbox, output_height, output_width)

    # Resize target backgrounds to foreground size
    target_img = cv2.resize(target_img, (output_width, output_height))
    target_green_img = cv2.resize(target_green_img, (output_width, output_height))
    # Compose cutout foreground on Background
    compose_target_img = composite4(fg_out, target_img, alpha_out)
    compose_target_green_img = composite4(fg_out, target_green_img, alpha_out)

    # Write results to videos
    masks_video_writer.write(np.repeat(np.expand_dims(alpha_out, -1), 3, axis=2))
    video_writer.write(cv2.cvtColor(compose_target_green_img, cv2.COLOR_BGR2RGB))

video_writer.release()
video.release()
print("Finished")
