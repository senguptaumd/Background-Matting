from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


import os
import time
import argparse
import numpy as np

from data_loader import VideoData
from functions import *
from networks import ResnetConditionHR, MultiscaleDiscriminator, conv_init
from loss_functions import alpha_loss, compose_loss, alpha_gradient_loss, GANloss


#CUDA

#os.environ["CUDA_VISIBLE_DEVICES"]="4"
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])


"""Parses arguments."""
parser = argparse.ArgumentParser(description='Training Background Matting on Adobe Dataset.')
parser.add_argument('-n', '--name', type=str, help='Name of tensorboard and model saving folders.')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch Size.')
parser.add_argument('-res', '--reso', type=int, help='Input image resolution')
parser.add_argument('-init_model', '--init_model', type=str, help='Initial model file')

parser.add_argument('-epoch', '--epoch', type=int, default=15,help='Maximum Epoch')
parser.add_argument('-n_blocks1', '--n_blocks1', type=int, default=7,help='Number of residual blocks after Context Switching.')
parser.add_argument('-n_blocks2', '--n_blocks2', type=int, default=3,help='Number of residual blocks for Fg and alpha each.')

args=parser.parse_args()

##Directories
tb_dir='TB_Summary/' + args.name
model_dir='Models/' + args.name


if not os.path.exists(model_dir):
	os.makedirs(model_dir)

if not os.path.exists(tb_dir):
	os.makedirs(tb_dir)

## Input list
data_config_train = {'reso': (args.reso,args.reso)}  #if trimap is true, rcnn is used

# DATA LOADING
print('\n[Phase 1] : Data Preparation')

def collate_filter_none(batch):
	batch = list(filter(lambda x: x is not None, batch))
	return torch.utils.data.dataloader.default_collate(batch)

#Original Data
traindata =  VideoData(csv_file='Video_data_train.csv',data_config=data_config_train,transform=None)  #Write a dataloader function that can read the database provided by .csv file
train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size, collate_fn=collate_filter_none)


print('\n[Phase 2] : Initialization')

netB=ResnetConditionHR(input_nc=(3,3,1,4),output_nc=4,n_blocks1=args.n_blocks1,n_blocks2=args.n_blocks2)
netB=nn.DataParallel(netB)
netB.load_state_dict(torch.load(args.init_model))
netB.cuda(); netB.eval()
for param in netB.parameters(): #freeze netD
	param.requires_grad = False

netG=ResnetConditionHR(input_nc=(3,3,1,4),output_nc=4,n_blocks1=args.n_blocks1,n_blocks2=args.n_blocks2)
netG.apply(conv_init)
netG=nn.DataParallel(netG)
netG.cuda()
torch.backends.cudnn.benchmark=True

netD=MultiscaleDiscriminator(input_nc=3,num_D=1,norm_layer=nn.InstanceNorm2d,ndf=64)
netD.apply(conv_init)
netD=nn.DataParallel(netD)
netD.cuda()

#Loss
l1_loss=alpha_loss()
c_loss=compose_loss()
g_loss=alpha_gradient_loss()
GAN_loss=GANloss()

optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
optimizerD = optim.Adam(netD.parameters(), lr=1e-5)



log_writer=SummaryWriter(tb_dir)

print('Starting Training')
step=50

KK=len(train_loader)

wt=1
for epoch in range(0,args.epoch):

	netG.train(); netD.train()

	lG, lD, GenL, DisL_r, DisL_f, alL, fgL, compL, elapse_run, elapse=0,0,0,0,0,0,0,0,0,0

	t0=time.time();


	for i,data in enumerate(train_loader):
		#Initiating

		bg, image, seg, multi_fr, seg_gt, back_rnd =  data['bg'], data['image'], data['seg'], data['multi_fr'], data['seg-gt'], data['back-rnd']

		bg, image, seg, multi_fr, seg_gt, back_rnd = Variable(bg.cuda()), Variable(image.cuda()), Variable(seg.cuda()), Variable(multi_fr.cuda()), Variable(seg_gt.cuda()), Variable(back_rnd.cuda())

		mask0=Variable(torch.ones(seg.shape).cuda())

		tr0=time.time()

		#pseudo-supervision
		alpha_pred_sup,fg_pred_sup=netB(image,bg,seg,multi_fr)
		mask=(alpha_pred_sup>-0.98).type(torch.cuda.FloatTensor)

		mask1=(seg_gt>0.95).type(torch.cuda.FloatTensor)

		## Train Generator

		alpha_pred,fg_pred=netG(image,bg,seg,multi_fr)

		##pseudo-supervised losses
		al_loss=l1_loss(alpha_pred_sup,alpha_pred,mask0)+0.5*g_loss(alpha_pred_sup,alpha_pred,mask0)
		fg_loss=l1_loss(fg_pred_sup,fg_pred,mask)

		#compose into same background
		comp_loss= c_loss(image,alpha_pred,fg_pred,bg,mask1)

		#randomly permute the background
		perm=torch.LongTensor(np.random.permutation(bg.shape[0]))
		bg_sh=bg[perm,:,:,:]

		al_mask=(alpha_pred>0.95).type(torch.cuda.FloatTensor)

		#Choose the target background for composition
		#back_rnd: contains separate set of background videos captured
		#bg_sh: contains randomly permuted captured background from the same minibatch
		if np.random.random_sample() > 0.5:
			bg_sh=back_rnd

		image_sh=compose_image_withshift(alpha_pred,image*al_mask + fg_pred*(1-al_mask),bg_sh,seg) 

		fake_response=netD(image_sh)

		loss_ganG=GAN_loss(fake_response,label_type=True)

		lossG= loss_ganG + wt*(0.05*comp_loss+0.05*al_loss+0.05*fg_loss)

		optimizerG.zero_grad()

		lossG.backward()
		optimizerG.step()

		##Train Discriminator
		
		fake_response=netD(image_sh); real_response=netD(image)

		loss_ganD_fake=GAN_loss(fake_response,label_type=False)
		loss_ganD_real=GAN_loss(real_response,label_type=True)

		lossD=(loss_ganD_real+loss_ganD_fake)*0.5

		# Update discriminator for every 5 generator update
		if i%5 ==0:
			optimizerD.zero_grad()
			lossD.backward()
			optimizerD.step()


		lG += lossG.data
		lD += lossD.data
		GenL += loss_ganG.data
		DisL_r += loss_ganD_real.data
		DisL_f += loss_ganD_fake.data

		alL += al_loss.data
		fgL += fg_loss.data
		compL += comp_loss.data


		log_writer.add_scalar('Generator Loss', lossG.data, epoch*KK + i + 1)
		log_writer.add_scalar('Discriminator Loss', lossD.data, epoch*KK + i + 1)
		log_writer.add_scalar('Generator Loss: Fake', loss_ganG.data, epoch*KK + i + 1)
		log_writer.add_scalar('Discriminator Loss: Real', loss_ganD_real.data, epoch*KK + i + 1)
		log_writer.add_scalar('Discriminator Loss: Fake', loss_ganD_fake.data, epoch*KK + i + 1)

		log_writer.add_scalar('Generator Loss: Alpha', al_loss.data, epoch*KK + i + 1)
		log_writer.add_scalar('Generator Loss: Fg', fg_loss.data, epoch*KK + i + 1)
		log_writer.add_scalar('Generator Loss: Comp', comp_loss.data, epoch*KK + i + 1)



		t1=time.time()

		elapse +=t1 -t0
		elapse_run += t1-tr0
		t0=t1

		if i % step == (step-1):
			print('[%d, %5d] Gen-loss:  %.4f Disc-loss: %.4f Alpha-loss: %.4f Fg-loss: %.4f Comp-loss: %.4f Time-all: %.4f Time-fwbw: %.4f' %(epoch + 1, i + 1, lG/step,lD/step,alL/step,fgL/step,compL/step,elapse/step,elapse_run/step))
			lG, lD, GenL, DisL_r, DisL_f, alL, fgL, compL, elapse_run, elapse=0,0,0,0,0,0,0,0,0,0

			write_tb_log(image,'image',log_writer,i)
			write_tb_log(seg,'seg',log_writer,i)
			write_tb_log(alpha_pred_sup,'alpha-sup',log_writer,i)
			write_tb_log(alpha_pred,'alpha_pred',log_writer,i)
			write_tb_log(fg_pred_sup*mask,'fg-pred-sup',log_writer,i)
			write_tb_log(fg_pred*mask,'fg_pred',log_writer,i)

			#composition
			alpha_pred=(alpha_pred+1)/2
			comp=fg_pred*alpha_pred + (1-alpha_pred)*bg
			write_tb_log(comp,'composite-same',log_writer,i)
			write_tb_log(image_sh,'composite-diff',log_writer,i)

			
			del comp

		del mask, back_rnd, mask0, seg_gt, mask1, bg, alpha_pred, alpha_pred_sup, image, fg_pred_sup, fg_pred, seg, multi_fr,image_sh, bg_sh, fake_response, real_response, al_loss, fg_loss, comp_loss, lossG, lossD, loss_ganD_real, loss_ganD_fake, loss_ganG


	
	if (epoch%2 == 0):
		torch.save(netG.state_dict(), model_dir + 'netG_epoch_%d.pth' %(epoch))
		torch.save(optimizerG.state_dict(), model_dir + 'optimG_epoch_%d.pth' %(epoch))
		torch.save(netD.state_dict(), model_dir + 'netD_epoch_%d.pth' %(epoch))
		torch.save(optimizerD.state_dict(), model_dir + 'optimD_epoch_%d.pth' %(epoch))

		#Change weight every 2 epoch to put more stress on discriminator weight and less on pseudo-supervision
		wt=wt/2
