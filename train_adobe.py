from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


import os
import time
import argparse

from data_loader import AdobeDataAffineHR
from functions import *
from networks import ResnetConditionHR, conv_init
from loss_functions import alpha_loss, compose_loss, alpha_gradient_loss


#CUDA

#os.environ["CUDA_VISIBLE_DEVICES"]="4"
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])


"""Parses arguments."""
parser = argparse.ArgumentParser(description='Training Background Matting on Adobe Dataset.')
parser.add_argument('-n', '--name', type=str, help='Name of tensorboard and model saving folders.')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch Size.')
parser.add_argument('-res', '--reso', type=int, help='Input image resolution')

parser.add_argument('-epoch', '--epoch', type=int, default=60,help='Maximum Epoch')
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
data_config_train = {'reso': [args.reso,args.reso], 'trimapK': [5,5], 'noise': True}  # choice for data loading parameters

# DATA LOADING
print('\n[Phase 1] : Data Preparation')

def collate_filter_none(batch):
	batch = list(filter(lambda x: x is not None, batch))
	return torch.utils.data.dataloader.default_collate(batch)

#Original Data
traindata =  AdobeDataAffineHR(csv_file='Data_adobe/Adobe_train_data.csv',data_config=data_config_train,transform=None)  #Write a dataloader function that can read the database provided by .csv file

train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size, collate_fn=collate_filter_none)


print('\n[Phase 2] : Initialization')

net=ResnetConditionHR(input_nc=(3,3,1,4), output_nc=4, n_blocks1=7, n_blocks2=3, norm_layer=nn.BatchNorm2d)
net.apply(conv_init)
net=nn.DataParallel(net)
#net.load_state_dict(torch.load(model_dir + 'net_epoch_X')) #uncomment this if you are initializing your model
net.cuda()
torch.backends.cudnn.benchmark=True

#Loss
l1_loss=alpha_loss()
c_loss=compose_loss()
g_loss=alpha_gradient_loss()

optimizer = optim.Adam(net.parameters(), lr=1e-4)
#optimizer.load_state_dict(torch.load(model_dir + 'optim_epoch_X')) #uncomment this if you are initializing your model



log_writer=SummaryWriter(tb_dir)

print('Starting Training')
step=50 #steps to visualize training images in tensorboard

KK=len(train_loader)

for epoch in range(0,args.epoch):

	net.train(); 

	netL, alL, fgL, fg_cL, al_fg_cL, elapse_run, elapse=0,0,0,0,0,0,0

	t0=time.time();
	testL=0; ct_tst=0;
	for i,data in enumerate(train_loader):
		#Initiating

		fg, bg, alpha, image, seg, bg_tr, multi_fr = data['fg'], data['bg'], data['alpha'], data['image'], data['seg'], data['bg_tr'], data['multi_fr']

		fg, bg, alpha, image, seg, bg_tr, multi_fr = Variable(fg.cuda()), Variable(bg.cuda()), Variable(alpha.cuda()), Variable(image.cuda()), Variable(seg.cuda()), Variable(bg_tr.cuda()), Variable(multi_fr.cuda())


		mask=(alpha>-0.99).type(torch.cuda.FloatTensor)
		mask0=Variable(torch.ones(alpha.shape).cuda())

		tr0=time.time()

		alpha_pred,fg_pred=net(image,bg_tr,seg,multi_fr)

		## Put needed loss here
		al_loss=l1_loss(alpha,alpha_pred,mask0)
		fg_loss=l1_loss(fg,fg_pred,mask)

		al_mask=(alpha_pred>0.95).type(torch.cuda.FloatTensor)
		fg_pred_c=image*al_mask + fg_pred*(1-al_mask)
		
		fg_c_loss= c_loss(image,alpha_pred,fg_pred_c,bg,mask0)

		al_fg_c_loss=g_loss(alpha,alpha_pred,mask0)

		loss=al_loss + 2*fg_loss + fg_c_loss + al_fg_c_loss

		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

		netL += loss.data
		alL += al_loss.data
		fgL += fg_loss.data
		fg_cL += fg_c_loss.data
		al_fg_cL += al_fg_c_loss.data


		log_writer.add_scalar('training_loss', loss.data, epoch*KK + i + 1)
		log_writer.add_scalar('alpha_loss', al_loss.data, epoch*KK + i + 1)
		log_writer.add_scalar('fg_loss', fg_loss.data, epoch*KK + i + 1)
		log_writer.add_scalar('comp_loss', fg_c_loss.data, epoch*KK + i + 1)
		log_writer.add_scalar('alpha_gradient_loss', al_fg_c_loss.data, epoch*KK + i + 1)


		t1=time.time()

		elapse +=t1 -t0
		elapse_run += t1-tr0

		t0=t1

		testL+=loss.data
		ct_tst+=1

		if i % step == (step-1):
			print('[%d, %5d] Total-loss:  %.4f Alpha-loss: %.4f Fg-loss: %.4f Comp-loss: %.4f Alpha-gradient-loss: %.4f Time-all: %.4f Time-fwbw: %.4f' % (epoch + 1, i + 1, netL/step, alL/step, fgL/step, fg_cL/step, al_fg_cL/step, elapse/step, elapse_run/step))
			netL, alL, fgL, fg_cL, al_fg_cL, elapse_run, elapse=0,0,0,0,0,0,0

			write_tb_log(image,'image',log_writer,i)
			write_tb_log(seg,'seg',log_writer,i)
			write_tb_log(alpha,'alpha',log_writer,i)
			write_tb_log(alpha_pred,'alpha_pred',log_writer,i)
			write_tb_log(fg*mask,'fg',log_writer,i)
			write_tb_log(fg_pred*mask,'fg_pred',log_writer,i)
			write_tb_log(multi_fr[0:4,0,...].unsqueeze(1),'multi_fr',log_writer,i)

			#composition
			alpha_pred=(alpha_pred+1)/2
			comp=fg_pred*alpha_pred + (1-alpha_pred)*bg
			write_tb_log(comp,'composite',log_writer,i)

			del comp

		del fg, bg, alpha, image, alpha_pred, fg_pred, seg, multi_fr


	#Saving
	torch.save(net.state_dict(), model_dir + 'net_epoch_%d_%.4f.pth' %(epoch,testL/ct_tst))
	torch.save(optimizer.state_dict(), model_dir + 'optim_epoch_%d_%.4f.pth' %(epoch,testL/ct_tst))


