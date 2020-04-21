import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
import pdb
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
#import scipy.io as sio


class alpha_loss(_Loss):
	def __init__(self):
		super(alpha_loss,self).__init__()

	def forward(self,alpha,alpha_pred,mask):
		return normalized_l1_loss(alpha,alpha_pred,mask)

		


class compose_loss(_Loss):
	def __init__(self):
		super(compose_loss,self).__init__()

	def forward(self,image,alpha_pred,fg,bg,mask):

		alpha_pred=(alpha_pred+1)/2

		comp=fg*alpha_pred + (1-alpha_pred)*bg

		return normalized_l1_loss(image,comp,mask)

class alpha_gradient_loss(_Loss):
	def __init__(self):
		super(alpha_gradient_loss,self).__init__()

	def forward(self,alpha,alpha_pred,mask):

		fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]); fx=fx.view((1,1,3,3)); fx=Variable(fx.cuda())
		fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]); fy=fy.view((1,1,3,3)); fy=Variable(fy.cuda())

		G_x = F.conv2d(alpha,fx,padding=1); G_y = F.conv2d(alpha,fy,padding=1)
		G_x_pred = F.conv2d(alpha_pred,fx,padding=1); G_y_pred = F.conv2d(alpha_pred,fy,padding=1)

		loss=normalized_l1_loss(G_x,G_x_pred,mask) + normalized_l1_loss(G_y,G_y_pred,mask)

		return loss

class alpha_gradient_reg_loss(_Loss):
	def __init__(self):
		super(alpha_gradient_reg_loss,self).__init__()

	def forward(self,alpha,mask):

		fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]); fx=fx.view((1,1,3,3)); fx=Variable(fx.cuda())
		fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]); fy=fy.view((1,1,3,3)); fy=Variable(fy.cuda())

		G_x = F.conv2d(alpha,fx,padding=1); G_y = F.conv2d(alpha,fy,padding=1)

		loss=(torch.sum(torch.abs(G_x))+torch.sum(torch.abs(G_y)))/torch.sum(mask)

		return loss


class GANloss(_Loss):
	def __init__(self):
		super(GANloss,self).__init__()

	def forward(self,pred,label_type):
		MSE=nn.MSELoss()

		loss=0
		for i in range(0,len(pred)):
			if label_type:
				labels=torch.ones(pred[i][0].shape)
			else:
				labels=torch.zeros(pred[i][0].shape)
			labels=Variable(labels.cuda())

			loss += MSE(pred[i][0],labels)

		return loss/len(pred)



def normalized_l1_loss(alpha,alpha_pred,mask):
	loss=0; eps=1e-6;
	for i in range(alpha.shape[0]):
		if mask[i,...].sum()>0:
			loss = loss + torch.sum(torch.abs(alpha[i,...]*mask[i,...]-alpha_pred[i,...]*mask[i,...]))/(torch.sum(mask[i,...])+eps)
	loss=loss/alpha.shape[0]
	
	return loss