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
#Set your paths here

#path to provided foreground images
fg_path = 'fg_train/'

#path to provided alpha mattes
a_path = 'mask_train/'

#Path to background images (MSCOCO)
bg_path = 'bg_train/'

#Path to folder where you want the composited images to go
out_path = 'merged_train/'

#csv file path
file_id= open("Adobe_train_data.csv","wb")

##############################################################

from PIL import Image
import os, glob
import math,random
import time, pdb

def composite4(fg, bg, a, w, h):
    
    bbox = fg.getbbox()
    bg = bg.crop((0,0,w,h))
    
    fg_list = fg.load()
    bg_list = bg.load()
    a_list = a.load()
    
    for y in range(h):
        for x in range (w):
            try:
                alpha = a_list[x,y] / 255
            except:
                alpha=(a_list[x,y][0])/255
            t = fg_list[x,y][0]
            t2 = bg_list[x,y][0]
            if alpha >= 1:
                r = int(fg_list[x,y][0])
                g = int(fg_list[x,y][1])
                b = int(fg_list[x,y][2])
                bg_list[x,y] = (r, g, b, 255)
            elif alpha > 0:
                r = int(alpha * fg_list[x,y][0] + (1-alpha) * bg_list[x,y][0])
                g = int(alpha * fg_list[x,y][1] + (1-alpha) * bg_list[x,y][1])
                b = int(alpha * fg_list[x,y][2] + (1-alpha) * bg_list[x,y][2])
                bg_list[x,y] = (r, g, b, 255)

    return bg

num_bgs = 100

fg_files = glob.glob(fg_path)
a_files = os.listdir(a_path)
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)
fail_list=[];
for im_name in fg_files:
    
    im_name=im_name.replace(fg_path,'')
    im = Image.open(fg_path + im_name);
    al = Image.open(a_path + im_name);
    bbox = im.size
    w = bbox[0]
    h = bbox[1]
    
    if im.mode != 'RGB' and im.mode != 'RGBA':
        im = im.convert('RGB')
    
    bcount = 0 
    for i in range(num_bgs):

        bg_name = next(bg_iter)        
        bg = Image.open(bg_path + bg_name)
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

        try:
            out = composite4(im, bg, al, w, h)
            out_name=out_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '_comp.png'
            out.save(out_name, "PNG")

            bbox = im.getbbox()
            back = bg.crop((0,0,w,h)) 
            back_name= out_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '_back.png'
            back.save(back_name, "PNG")

            #write to file
            line='Data_adobe/' + fg_path + im_name + ';' + 'Data_adobe/' + a_path + im_name + ';' + 'Data_adobe/' + out_name + ';' + 'Data_adobe/' + back_name + '\n'
            file_id.write(line)

        except:
            fail_list.append(im_name)


         

       
        bcount += 1

    print('Done: ' + im_name)

file_id.close()

