##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.


##############################################################
# python Composition_code.py --fg_path fg_train --mask_path mask_train --bg_path bg_train --out_path merged_train --out_csv Adobe_train_data.csv
# python Composition_code.py --fg_path fg_train --mask_path mask_train --bg_path /media/andrey/Elements2/COCO/COCO/train2017 --out_path /media/andrey/Elements2/RESEARCH/adobe/merged_train --out_csv Adobe_train_data.csv --workers 6

from PIL import Image
from tqdm import tqdm
import argparse
import os
import math
from multiprocessing.pool import ThreadPool
import threading

parser = argparse.ArgumentParser(description='compose backgrounds and foregrounds')

parser.add_argument('--fg_path', type=str, required=True, help='path to provided foreground images')
parser.add_argument('--mask_path', type=str, required=True, help='path to provided alpha mattes')
parser.add_argument('--bg_path', type=str, required=True, help='path to to background images (MSCOCO)')
parser.add_argument('--out_path', type=str, required=True, help='path to folder where you want the composited images to go')

parser.add_argument('--out_csv', type=str, default=os.devnull, help='path to csv file used by data loader')
parser.add_argument('--num_bgs', type=int, default=100, help='number of backgrounds onto which to paste each foreground')
parser.add_argument('--workers', type=int, help='maximum workers to use, defaults to os.cpu_count()')
args = parser.parse_args()

fg_path, a_path, bg_path, out_path, num_bgs = args.fg_path, args.mask_path, args.bg_path, args.out_path, args.num_bgs
os.makedirs(out_path, exist_ok=True)

def composite4(fg, bg, a, w, h):
    bg = bg.crop((0,0,w,h))
    bg.paste(fg, mask=a)
    return bg

def process_foreground_image(i, job):
    worker_thread_id = int(threading.current_thread().name.rpartition("-")[-1])
    im_name, bg_batch = job

    im_name = im_name.replace(fg_path, '')
    im = Image.open(os.path.join(fg_path, im_name))
    al = Image.open(os.path.join(a_path, im_name))
    bbox = im.size
    w = bbox[0]
    h = bbox[1]
    if im.mode != 'RGB' and im.mode != 'RGBA':
        im = im.convert('RGB')

    output_lines = []
    pretty_name = ("..." + im_name[-27:] if len(im_name) > 30 else im_name).rjust(30)
    with lock:
        pbar = tqdm(bg_batch, position=worker_thread_id, desc=f"({i}) {pretty_name}", leave=False)
    for b, bg_name in enumerate(pbar):
        bg = Image.open(os.path.join(bg_path, bg_name))
        if bg.mode != 'RGB':
            bg = bg.convert('RGB')

        bg_bbox = bg.size
        bw = bg_bbox[0]
        bh = bg_bbox[1]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = bg.resize((math.ceil(bw * ratio), math.ceil(bh * ratio)), Image.BICUBIC)

        try:
            out = composite4(im, bg, al, w, h)
            back_idx = i * num_bgs + b
            out_name = os.path.join(out_path, im_name[:len(im_name) - 4] + '_' + str(back_idx) + '_comp.png')
            out.save(out_name, "PNG")

            back = bg.crop((0, 0, w, h))
            back_name = os.path.join(out_path, im_name[:len(im_name) - 4] + '_' + str(back_idx) + '_back.png')
            back.save(back_name, "PNG")

            line = 'Data_adobe/' + os.path.join(fg_path, im_name) + ';' + 'Data_adobe/' + os.path.join(a_path, im_name) + ';' + 'Data_adobe/' + out_name + ';' + 'Data_adobe/' + back_name + '\n'
            output_lines.append(line)
        except Exception as e:
            tqdm.write(f"Composing {im_name} onto {bg_name} failed! Skipping...", e)
        with lock:
            pbar.update()
    with lock:
        pbar.close()
    return output_lines


fg_files = os.listdir(fg_path)[:20]
a_files = os.listdir(a_path)
bg_files = os.listdir(bg_path)
bg_batches = [bg_files[i * num_bgs:(i + 1) * num_bgs] for i in range((len(bg_files) + num_bgs - 1) // num_bgs )]


lock = threading.Lock()
pool = ThreadPool(args.workers)
with lock:
    total_pbar = tqdm(total=len(fg_files), position=args.workers+1, desc="TOTAL", leave=False, smoothing=0.0)
def update_total_pbar(_):
    with lock:
        total_pbar.update(1)
jobs = []
for jobargs in enumerate(zip(fg_files, bg_batches)):
    jobs.append(pool.apply_async(process_foreground_image, args=jobargs, callback=update_total_pbar))
pool.close()
pool.join()

output = []
for result in jobs:
    output.extend(result.get())
tqdm.write("Done composing...")

with open(args.out_csv, "w") as f:
    for line in output:
        f.write(line)
