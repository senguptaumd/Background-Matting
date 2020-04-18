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
# python Composition_code.py --fg_path fg_train --mask_path mask_train --bg_path /media/andrey/Elements2/COCO/COCO/train2017 --out_path merged_train --out_csv Adobe_train_data.csv

from PIL import Image
from tqdm import tqdm
import argparse
import os
import math
import numpy as np

parser = argparse.ArgumentParser(description='compose backgrounds and foregrounds')
parser.add_argument('--fg_path', type=str, required=True, help='path to provided foreground images')
parser.add_argument('--mask_path', type=str, required=True, help='path to provided alpha mattes')
parser.add_argument('--bg_path', type=str, required=True, help='path to to background images (MSCOCO)')
parser.add_argument('--out_path', type=str, required=True, help='path to folder where you want the composited images to go')
parser.add_argument('--out_csv', type=str, default=os.devnull, help='path to csv file used by data loader')
args = parser.parse_args()

fg_path, a_path, bg_path, out_path = args.fg_path, args.mask_path, args.bg_path, args.out_path
os.makedirs(out_path, exist_ok=True)
file_id = open(args.out_csv, "w")

def composite4(fg, bg, a, w, h):

    bg = bg.crop((0,0,w,h))

    arr_bg = np.array(bg)
    assert arr_bg.ndim == 3
    arr_fg = np.array(fg)
    arr_a = np.array(a)
    if arr_a.ndim == 1:
        arr_a = arr_a[:, :, 0]
    arr_a = np.divide(arr_a, 255.0)[..., None]
    arr_blend = arr_a * arr_fg + (1 - arr_a) * arr_bg
    arr_blend = arr_blend.astype(np.uint8)
    result = Image.fromarray(arr_blend)



    
    # fg_list = fg.load()
    # bg_list = bg.load()
    # a_list = a.load()
    #
    # for y in range(h):
    #     for x in range (w):
    #         try:
    #             alpha = a_list[x,y] / 255
    #         except:
    #             alpha=(a_list[x,y][0])/255
    #         # t = fg_list[x,y][0]
    #         # t2 = bg_list[x,y][0]
    #         if alpha >= 1:
    #             r = int(fg_list[x,y][0])
    #             g = int(fg_list[x,y][1])
    #             b = int(fg_list[x,y][2])
    #             bg_list[x,y] = (r, g, b, 255)
    #         elif alpha > 0:
    #             r = int(alpha * fg_list[x,y][0] + (1-alpha) * bg_list[x,y][0])
    #             g = int(alpha * fg_list[x,y][1] + (1-alpha) * bg_list[x,y][1])
    #             b = int(alpha * fg_list[x,y][2] + (1-alpha) * bg_list[x,y][2])
    #             bg_list[x,y] = (r, g, b, 255)
    # assert np.array_equal(np.array(result), np.array(bg))
    # print("Assert passed.")

    # return bg
    return result

num_bgs = 100

fg_files = os.listdir(fg_path)
a_files = os.listdir(a_path)
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)
for im_name in tqdm(fg_files):
    
    im_name=im_name.replace(fg_path,'')
    im = Image.open(os.path.join(fg_path, im_name))
    al = Image.open(os.path.join(a_path, im_name))
    bbox = im.size
    w = bbox[0]
    h = bbox[1]
    
    if im.mode != 'RGB' and im.mode != 'RGBA':
        im = im.convert('RGB')
    
    bcount = 0 
    for i in tqdm(range(num_bgs), leave=False):

        bg_name = next(bg_iter)        
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
            bg = bg.resize((math.ceil(bw*ratio),math.ceil(bh*ratio)), Image.BICUBIC)

        out = composite4(im, bg, al, w, h)
        out_name = os.path.join(out_path, im_name[:len(im_name)-4] + '_' + str(bcount) + '_comp.png')
        out.save(out_name, "PNG")

        bbox = im.getbbox()
        back = bg.crop((0,0,w,h))
        back_name = os.path.join(out_path, im_name[:len(im_name)-4] + '_' + str(bcount) + '_back.png')
        back.save(back_name, "PNG")

        #write to file
        line = 'Data_adobe/' + os.path.join(fg_path, im_name) + ';' + 'Data_adobe/' + os.path.join(a_path, im_name) + ';' + 'Data_adobe/' + out_name + ';' + 'Data_adobe/' + back_name + '\n'
        file_id.write(line)


        bcount += 1

    print('Done: ' + im_name)

file_id.close()

