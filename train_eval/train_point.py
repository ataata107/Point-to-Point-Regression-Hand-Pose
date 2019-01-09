'''
training
author: Liuhao Ge
'''
import argparse
import os
import random
import progressbar
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset_point import HandPointDataset
from dataset_point import subject_names
from dataset_point import gesture_names
from network_point import PointNet_Plus
from utils_point import group_points,offset_cal

#print(torch.cuda.current_device())
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=-1, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (SGD only)')
parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 6,  help='number of input point features')
parser.add_argument('--PCA_SZ', type=int, default = 42,  help='number of PCA components')
parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

parser.add_argument('--test_index', type=int, default = 0,  help='test index for cross validation, range: 0~8')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')

opt = parser.parse_args()
print (opt)

torch.cuda.set_device(opt.main_gpu)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])

try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
train_data = HandPointDataset(root_path='../preprocess', opt=opt, train = True)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
										  shuffle=True, num_workers=int(opt.workers), pin_memory=False)
										  
test_data = HandPointDataset(root_path='../preprocess', opt=opt, train = False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)
										  
print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
netR = PointNet_Plus(opt)
if opt.ngpu > 1:
	netR.netR_1 = torch.nn.DataParallel(netR.netR_1, range(opt.ngpu))
	netR.netR_2 = torch.nn.DataParallel(netR.netR_2, range(opt.ngpu))
	netR.netR_3 = torch.nn.DataParallel(netR.netR_3, range(opt.ngpu))
if opt.model != '':
	netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
	
netR.cpu()
print(netR)

criterion1 = nn.MSELoss(size_average=True).cpu()
optimizer1 = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)

criterion2 = nn.MSELoss(size_average=True).cpu()
optimizer2 = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.1)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.1)

# 3. Training and testing
for epoch in range(opt.nepoch):
	scheduler1.step(epoch)
	scheduler2.step(epoch)
	print('======>>>>> Online epoch: #%d, lr=%f, Test: %s <<<<<======' %(epoch, scheduler1.get_lr()[0], subject_names[opt.test_index]))
	# 3.1 switch to train mode
	#torch.cuda.synchronize()
	netR.train()
	train_mse = 0.0
	train_mse_wld = 0.0
	timer = time.time()

	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data[0]) == 1:
			continue
		#torch.cuda.synchronize()       
		# 3.1.1 load inputs and targets
		points, volume_length, gt_pca, gt_xyz,jnts = data
		gt_pca = Variable(gt_pca, requires_grad=False).cpu()
		points, volume_length, gt_xyz = points.cpu(), volume_length.cpu(), gt_xyz.cpu()
		
		#load offsets and heatmaps
		points_htm=points[:,:,:3]
		jnts=torch.cat((jnts,points_htm),1)
		heatmap = offset_cal(jnts,opt)  #B*4J*1024*1
		

		# points: B * 1024 * 6; target: B * 42
		inputs_level1, inputs_level1_center = group_points(points, opt)
		inputs_level1, inputs_level1_center = Variable(inputs_level1, requires_grad=False), Variable(inputs_level1_center, requires_grad=False) #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1

		# 3.1.2 compute output
		optimizer1.zero_grad()
		optimizer2.zero_grad()
		points=points.permute(0,2,1).unsqueeze(3) # points: B * 6 * 1024 * 1
		estimation1,estimation2 = netR(inputs_level1, inputs_level1_center,points)
		
		loss1 = criterion1(estimation1, heatmap)
		loss2 = criterion2(estimation2, heatmap)
		# 3.1.3 compute gradient and do SGD step
		loss1.backward()
		loss2.backward()
		optimizer1.step()
		optimizer2.step()
		#torch.cuda.synchronize()
		
		# 3.1.4 update training error
		train_mse = train_mse + (loss1.data[0]+loss2.data[0])*len(points)/2.0
		
		
		
	# time taken
	#torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

	# print mse
	train_mse = train_mse / len(train_data)
	
	print('mean-square error of 1 sample: %f, #train_data = %d' %(train_mse, len(train_data)))
	

	torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
	torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))
	
