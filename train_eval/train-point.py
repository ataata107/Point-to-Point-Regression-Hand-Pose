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

from dataset import HandPointDataset
from dataset import subject_names
from dataset import gesture_names
from network import PointNet_Plus
from utils import group_points

from pointnet2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

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
classifier = PointNet2ClsSsg()
classifier.cuda()
print(netR)	


criterion = nn.MSELoss(size_average=True).cuda()
optimizer = optim.Adam(classifier.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)								  
									

for epoch in range(opt.nepoch):
    scheduler.step(epoch)
	print('======>>>>> Online epoch: #%d, lr=%f, Test: %s <<<<<======' %(epoch, scheduler.get_lr()[0], subject_names[opt.test_index]))
	# 3.1 switch to train mode
	torch.cuda.synchronize()
	netR.train()
	train_mse = 0.0
	train_mse_wld = 0.0
	timer = time.time()
	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()       
		# 3.1.1 load inputs and targets
		points, volume_length, gt_pca, gt_xyz = data
		gt_pca = Variable(gt_pca, requires_grad=False).cuda()
		points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
		
		#Output heatmaps
		heatmap,heatmap_vector = group_points(gt_xyz, opt)

		# points: B * 1024 * 6; target: B * 42
		inputs_level1, inputs_level1_center = group_points(points, opt)
		inputs_level1, inputs_level1_center = Variable(inputs_level1,	


    print("Train examples: {}".format(train_examples))
    print("Evaluation examples: {}".format(test_examples))
    print("Start training...")
    cudnn.benchmark = True
    classifier.cuda()
    
        print("--------Epoch {}--------".format(epoch))

        # train one epoch
        classifier.train()
        total_train_loss = 0
        correct_examples = 0
        for batch_idx, data in enumerate(train_dataloader, 0):
            pointcloud, label = data
            pointcloud = pointcloud.permute(0, 2, 1)
            pointcloud, label = pointcloud.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = classifier(pointcloud)

            loss = F.nll_loss(pred, label.view(-1))
            pred_choice = pred.max(1)[1]

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_examples += pred_choice.eq(label.view(-1)).sum().item()
            
        print("Train loss: {:.4f}, train accuracy: {:.2f}%".format(total_train_loss / train_batches, correct_examples / train_examples * 100.0))

        # eval one epoch
        classifier.eval()
        correct_examples = 0
        for batch_idx, data in enumerate(test_dataloader, 0):
            pointcloud, label = data
            pointcloud = pointcloud.permute(0, 2, 1)
            pointcloud, label = pointcloud.cuda(), label.cuda()

            pred = classifier(pointcloud)
            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(label.view(-1)).sum()
            correct_examples += correct.item()

        print("Eval accuracy: {:.2f}%".format(correct_examples / test_examples * 100.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pointnet Trainer')
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=10)
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
    args = parser.parse_args()
    train(args)
    