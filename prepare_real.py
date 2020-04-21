#######################################
# Prepares training data. Takes a path to a directory of videos + captured backgrounds, dumps frames, extracts human
# segmentations. Also takes a path of background videos. Creates a training CSV file with lines of the following format,
# by using all but the last 80 frames of each video and iterating repeatedly over the background frames as needed.

#$image;$captured_back;$segmentation;$image+20frames;$image+2*20frames;$image+3*20frames;$image+4*20frames;$target_back

path = "/path/to/Captured_Data/fixed-camera/train"
background_path = "/path/to/Captured_Data/background"
output_csv = "Video_data_train.csv"

#######################################

import os
from itertools import cycle
from tqdm import tqdm

videos = [os.path.join(path, f[:-4]) for f in os.listdir(path) if f.endswith(".mp4")]
backgrounds = [os.path.join(background_path, f[:-4]) for f in os.listdir(background_path) if f.endswith(".MOV")]

print(f"Dumping frames and segmenting {len(videos)} input videos")
for i, video in enumerate(tqdm(videos)):
    os.makedirs(video, exist_ok=True)
    code = os.system(f"ffmpeg -i {video}.mp4 {video}/%04d_img.png -hide_banner > prepare_real_logs.txt 2>&1")
    if code != 0:
        exit(code)
    print(f"Dumped frames for {video} ({i+1}/{len(videos)})")
    code = os.system(f"python test_segmentation_deeplab.py -i {video} > prepare_real_logs.txt 2>&1")
    if code != 0:
        exit(code)
    print(f"Segmented {video} ({i+1}/{len(videos)})")

print(f"Dumping frames for {background_path} background videos")
for i, background in enumerate(tqdm(backgrounds)):
    os.makedirs(background, exist_ok=True)
    code = os.system(f"ffmpeg -i {background}.MOV {background}/%04d_img.png -hide_banner > /dev/null 2>&1")
    if code != 0:
        exit(code)
    print(f"Dumped frames for {background} ({i+1}/{len(videos)})")

print(f"Creating CSV")
background_frames = []
for background in backgrounds:
    background_frames.extend([os.path.join(background, f) for f in sorted(os.listdir(background))])
background_stream = cycle(background_frames)

with open(output_csv, "w") as f:
    for i, video in enumerate(videos):
        n = len(os.listdir(video))
        assert n % 2 == 0
        n //= 2
        for j in range(1, n + 1 - 80):
            img_name = video + "/%04d_img.png" % j
            captured_back = video + ".png"
            seg_name = video + "/%04d_masksDL.png" % j
            mc1 = video + "/%04d_img.png" % (j + 20)
            mc2 = video + "/%04d_img.png" % (j + 40)
            mc3 = video + "/%04d_img.png" % (j + 60)
            mc4 = video + "/%04d_img.png" % (j + 80)
            target_back = next(background_stream)
            csv_line = f"{img_name};{captured_back};{seg_name};{mc1};{mc2};{mc3};{mc4};{target_back}\n"
            f.write(csv_line)

print(f"Done, written to {output_csv}")
