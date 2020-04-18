from __future__ import print_function, division
import os
import torch
import pandas as pd
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pdb, random
from torch.utils.data import Dataset, DataLoader
import random, os, cv2

unknown_code=128

class VideoData(Dataset):
	def __init__(self,csv_file,data_config,transform=None):
		self.frames = pd.read_csv(csv_file,sep=';')
		self.transform = transform
		self.resolution=data_config['reso']
		
	def __len__(self):
		return len(self.frames)

	def __getitem__(self,idx):
		img = io.imread(self.frames.iloc[idx, 0])
		back = io.imread(self.frames.iloc[idx, 1])
		seg = io.imread(self.frames.iloc[idx, 2])
		
		fr1 = cv2.cvtColor(io.imread(self.frames.iloc[idx, 3]), cv2.COLOR_BGR2GRAY)
		fr2 = cv2.cvtColor(io.imread(self.frames.iloc[idx, 4]), cv2.COLOR_BGR2GRAY)
		fr3 = cv2.cvtColor(io.imread(self.frames.iloc[idx, 5]), cv2.COLOR_BGR2GRAY)
		fr4 = cv2.cvtColor(io.imread(self.frames.iloc[idx, 6]), cv2.COLOR_BGR2GRAY)

		back_rnd = io.imread(self.frames.iloc[idx, 7])
		
		sz=self.resolution

		if np.random.random_sample() > 0.5:
			img = cv2.flip(img,1)
			seg = cv2.flip(seg,1)
			back = cv2.flip(back,1)
			back_rnd = cv2.flip(back_rnd,1)
			fr1=cv2.flip(fr1,1); fr2=cv2.flip(fr2,1); fr3=cv2.flip(fr3,1); fr4=cv2.flip(fr4,1)

		#make frames together
		multi_fr=np.zeros((img.shape[0],img.shape[1],4))
		multi_fr[...,0]=fr1; multi_fr[...,1]=fr2; multi_fr[...,2]=fr3; multi_fr[...,3]=fr4;
			
		
		#allow random cropping centered on the segmentation map
		bbox=create_bbox(seg,seg.shape[0],seg.shape[1])
		img=apply_crop(img,bbox,self.resolution)
		seg=apply_crop(seg,bbox,self.resolution)
		back=apply_crop(back,bbox,self.resolution)
		back_rnd=apply_crop(back_rnd,bbox,self.resolution)
		multi_fr=apply_crop(multi_fr,bbox,self.resolution)

		#convert seg to guidance map
		#segg=create_seg_guide(seg,self.resolution)

		sample = {'image': to_tensor(img), 'seg': to_tensor(create_seg_guide(seg,self.resolution)), 'bg': to_tensor(back), 'multi_fr': to_tensor(multi_fr), 'seg-gt':to_tensor(seg), 'back-rnd': to_tensor(back_rnd)}

		if self.transform:
			sample = self.transform(sample)
		return sample


class AdobeDataAffineHR(Dataset):
	def __init__(self,csv_file,data_config,transform=None):
		self.frames = pd.read_csv(csv_file,sep=';')
		self.transform = transform
		self.resolution=data_config['reso']
		self.trimapK=data_config['trimapK']
		self.noise=data_config['noise']
		
	def __len__(self):
		return len(self.frames)

	def __getitem__(self,idx):
		try:
			#load
			fg = io.imread(self.frames.iloc[idx, 0])
			alpha = io.imread(self.frames.iloc[idx, 1])
			image = io.imread(self.frames.iloc[idx, 2])
			back = io.imread(self.frames.iloc[idx, 3])

			fg = cv2.resize(fg, dsize=(800,800))
			alpha = cv2.resize(alpha, dsize=(800,800))
			back = cv2.resize(back, dsize=(800,800))
			image = cv2.resize(image, dsize=(800,800))


			sz=self.resolution

			#random flip
			if np.random.random_sample() > 0.5:
				alpha = cv2.flip(alpha,1)
				fg = cv2.flip(fg,1)
				back = cv2.flip(back,1)
				image = cv2.flip(image,1)

			trimap=generate_trimap(alpha,self.trimapK[0],self.trimapK[1],False)


			#randcom crop+scale
			different_sizes = [(576,576),(608,608),(640,640),(672,672),(704,704),(736,736),(768,768),(800,800)]
			crop_size = random.choice(different_sizes)

			x, y = random_choice(trimap, crop_size)

			fg = safe_crop(fg, x, y, crop_size,sz)
			alpha = safe_crop(alpha, x, y, crop_size,sz)
			image = safe_crop(image, x, y, crop_size,sz)
			back = safe_crop(back, x, y, crop_size,sz)
			trimap = safe_crop(trimap, x, y, crop_size,sz)

			#Perturb Background: random noise addition or gamma change
			if self.noise:
				if np.random.random_sample() > 0.6:
					sigma=np.random.randint(low=2, high=6)
					mu=np.random.randint(low=0, high=14)-7
					back_tr=add_noise(back,mu,sigma)
				else:
					back_tr=skimage.exposure.adjust_gamma(back,np.random.normal(1,0.12))


			#Create motion cues: transform foreground and create 4 additional images
			affine_fr=np.zeros((fg.shape[0],fg.shape[1],4))
			for t in range(0,4):
				T=np.random.normal(0,5,(2,1)); theta=np.random.normal(0,7);
				R=np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],[np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
				sc=np.array([[1+np.random.normal(0,0.05), 0],[0,1]]); sh=np.array([[1, np.random.normal(0,0.05)*(np.random.random_sample() > 0.5)],[np.random.normal(0,0.05)*(np.random.random_sample() > 0.5), 1]]);
				A=np.concatenate((sc*sh*R, T), axis=1);

				fg_tr = cv2.warpAffine(fg.astype(np.uint8),A,(fg.shape[1],fg.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
				alpha_tr = cv2.warpAffine(alpha.astype(np.uint8),A,(fg.shape[1],fg.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)

				sigma=np.random.randint(low=2, high=6)
				mu=np.random.randint(low=0, high=14)-7
				back_tr0=add_noise(back,mu,sigma)

				affine_fr[...,t]=cv2.cvtColor(composite(fg_tr,back_tr0,alpha_tr), cv2.COLOR_BGR2GRAY)



			sample = {'image': to_tensor(image), 'fg': to_tensor(fg), 'alpha': to_tensor(alpha), 'bg': to_tensor(back), 'trimap': to_tensor(trimap), 'bg_tr': to_tensor(back_tr), 'seg': to_tensor(create_seg(alpha,trimap)), 'multi_fr': to_tensor(affine_fr)}



			if self.transform:
				sample = self.transform(sample)
			return sample
		except Exception as e:
			print("Error loading: " + self.frames.iloc[idx, 0])
			print(e)


#Functions

def create_seg_guide(rcnn,reso):
	kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	rcnn=rcnn.astype(np.float32)/255; rcnn[rcnn>0.2]=1;
	K=25

	zero_id=np.nonzero(np.sum(rcnn,axis=1)==0)
	del_id=zero_id[0][zero_id[0]>250]
	if len(del_id)>0:
		del_id=[del_id[0]-2,del_id[0]-1,*del_id]
		rcnn=np.delete(rcnn,del_id,0)
	rcnn = cv2.copyMakeBorder( rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)

	rcnn = cv2.erode(rcnn, kernel_er, iterations=np.random.randint(10,20))
	rcnn = cv2.dilate(rcnn, kernel_dil, iterations=np.random.randint(3,7))
	k_size_list=[(21,21),(31,31),(41,41)]
	rcnn=cv2.GaussianBlur(rcnn.astype(np.float32),random.choice(k_size_list),0)
	rcnn=(255*rcnn).astype(np.uint8)
	rcnn=np.delete(rcnn, range(reso[0],reso[0]+K), 0)

	return rcnn

def crop_holes(img,cx,cy,crop_size):
	img[cy:cy+crop_size[0],cx:cx+crop_size[1]]=0
	return img

def create_seg(alpha,trimap):
	#old
	num_holes=np.random.randint(low=0, high=3)
	crop_size_list=[(15,15),(25,25),(35,35),(45,45)]
	kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	seg = (alpha>0.5).astype(np.float32)
	#print('Before %.4f max: %.4f' %(seg.sum(),seg.max()))
	#old
	seg = cv2.erode(seg, kernel_er, iterations=np.random.randint(low=10,high=20))
	seg = cv2.dilate(seg, kernel_dil, iterations=np.random.randint(low=15,high=30))
	#print('After %.4f max: %.4f' %(seg.sum(),seg.max()))
	seg=seg.astype(np.float32)
	seg=(255*seg).astype(np.uint8)
	for i in range(num_holes):
		crop_size=random.choice(crop_size_list)
		cx,cy = random_choice(trimap,crop_size)
		seg=crop_holes(seg,cx,cy,crop_size)
		trimap=crop_holes(trimap,cx,cy,crop_size)
	k_size_list=[(21,21),(31,31),(41,41)]
	seg=cv2.GaussianBlur(seg.astype(np.float32),random.choice(k_size_list),0)
	return seg.astype(np.uint8)


def apply_crop(img,bbox,reso):
	img_crop=img[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3],...]; 
	img_crop=cv2.resize(img_crop,reso)
	return img_crop

def create_bbox(mask,R,C):
	where = np.array(np.where(mask))
	x1, y1 = np.amin(where, axis=1)
	x2, y2 = np.amax(where, axis=1)

	w=np.maximum(y2-y1,x2-x1);
	bd=np.random.uniform(0.1,0.4)
	x1=x1-np.round(bd*w)
	y1=y1-np.round(bd*w)
	y2=y2+np.round(bd*w)
	
	if x1<0: x1=0
	if y1<0: y1=0
	if y2>=C: y2=C
	if x2>=R: x2=R-1
	
	bbox=np.around([x1,y1,x2-x1,y2-y1]).astype('int')

	return bbox

def composite(fg, bg, a):
	fg = fg.astype(np.float32); bg=bg.astype(np.float32); a=a.astype(np.float32);
	alpha= np.expand_dims(a / 255,axis=2)
	im = alpha * fg + (1 - alpha) * bg
	im = im.astype(np.uint8)
	return im

def add_noise(back,mean,sigma):
	back=back.astype(np.float32)
	row,col,ch= back.shape
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	#gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
	noisy = back + gauss

	noisy[noisy<0]=0; noisy[noisy>255]=255;

	return noisy.astype(np.uint8)

def safe_crop(mat, x, y, crop_size,img_size,cubic=True):
	img_rows, img_cols = img_size
	crop_height, crop_width = crop_size
	if len(mat.shape) == 2:
		ret = np.zeros((crop_height, crop_width), np.float32)
	else:
		ret = np.zeros((crop_height, crop_width, 3), np.float32)
	crop = mat[y:y + crop_height, x:x + crop_width]
	h, w = crop.shape[:2]
	ret[0:h, 0:w] = crop
	if crop_size != (img_rows, img_cols):
		if cubic:
			ret = cv2.resize(ret, dsize=(img_rows, img_cols))
		else:
			ret = cv2.resize(ret, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
	return ret

def generate_trimap(alpha,K1,K2,train_mode):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	fg = np.array(np.equal(alpha, 255).astype(np.float32))
	if train_mode:
		K=np.random.randint(K1,K2)
	else:
		K=np.round((K1+K2)/2).astype('int')

	fg = cv2.erode(fg, kernel, iterations=K)
	unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	unknown = cv2.dilate(unknown, kernel, iterations=2*K)
	trimap = fg * 255 + (unknown - fg) * 128
	return trimap.astype(np.uint8)


def random_choice(trimap, crop_size=(320, 320)):
	img_height, img_width = trimap.shape[0:2]
	crop_height, crop_width = crop_size

	val_idx=np.zeros((img_height,img_width))
	val_idx[int(crop_height/2):int(img_height-crop_height/2),int(crop_width/2):int(img_width-crop_width/2)]=1

	y_indices, x_indices = np.where(np.logical_and(trimap == unknown_code,val_idx==1))
	num_unknowns = len(y_indices)
	x, y = 0, 0
	if num_unknowns > 0:
		ix = np.random.choice(range(num_unknowns))
		center_x = x_indices[ix]
		center_y = y_indices[ix]
		x = max(0, center_x - int(crop_width / 2))
		y = max(0, center_y - int(crop_height / 2))

	#added extra

	return x, y

def to_tensor(pic):
	if len(pic.shape)>=3:
		img = torch.from_numpy(pic.transpose((2, 0, 1)))
	else:
		img=torch.from_numpy(pic)
		img=img.unsqueeze(0)
	# backward compatibility


	return 2*(img.float().div(255))-1
