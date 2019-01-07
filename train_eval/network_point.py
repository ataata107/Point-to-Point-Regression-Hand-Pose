import torch
import torch.nn as nn
import math
from utils_point import group_points_2,group_points_3
nstates_plus_1 = [64,64,128]
nstates_plus_2 = [128,128,256]
nstates_plus_3 = [256,512,1024,1024,512]

def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def pointnetfc(xyz1, xyz2, points1, points2):
	xyz1=xyz1.squeeze(3)  #B*3*N1
	if xyz2 is not None:
		xyz2=xyz2.squeeze(3)		#B*3*N2
		xyz2 = xyz2.permute(0, 2, 1)
	points1=points1.squeeze(3)  #B*C1*N1
	points2=points2.squeeze(3)  #B*C2*N2
	
	xyz1 = xyz1.permute(0, 2, 1)
	
	points2 = points2.permute(0, 2, 1)
	B, N, C = xyz1.shape
	_, S, _ = xyz2.shape
	if S == 1:
		interpolated_points = points2.repeat(1, N, 1)
	else:
		dists = square_distance(xyz1, xyz2)
		dists, idx = dists.sort(dim=-1)
		dists, idx = dists[:,:,:3], idx[:,:,:3] #[B, N, 3]
		dists[dists < 1e-10] = 1e-10
		weight = 1.0 / dists #[B, N, 3]
		weight = weight / torch.sum(weight, dim=-1).view(B, N, 1) #[B, N, 3]
		interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim = 2)
	
	if points1 is not None:
		points1 = points1.permute(0, 2, 1)
		new_points = torch.cat([points1, interpolated_points], dim=-1)
	else:
		new_points = interpolated_points
		
	new_points = new_points.permute(0, 2, 1)
	return new_points
	
	

class PointNet_Plus(nn.Module):
    def __init__(self, opt):
        super(PointNet_Plus, self).__init__()
        self.num_outputs = opt.PCA_SZ
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        
        self.netR_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*128*sample_num_level1*1
        )
        self.netR_11 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(3+128+opt.JOINT_NUM*4, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*128*sample_num_level1*1
        )
        
        self.netR_2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )
        
        self.netR_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )
        self.netR_4 = nn.Sequential(
            # B*1280*sample_num_level2
            nn.Conv1d(1280, 256, kernel_size=(1, 1)),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2
            nn.Conv1d(256, 256, kernel_size=(1, 1)),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2
            
        )
        self.netR_5 = nn.Sequential(
            # B*384*sample_num_level1
            nn.Conv1d(384, 256, kernel_size=(1, 1)),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level1
            nn.Conv1d(256, 128, kernel_size=(1, 1)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1
            
        )
        self.netR_6 = nn.Sequential(
            # B*128*1024
            nn.Conv1d(128, 128, kernel_size=(1, 1)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # B*128*1024
            nn.Conv1d(128, 128, kernel_size=(1, 1)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # B*128*1024
            nn.Conv1d(128, 128, kernel_size=(1, 1)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # B*128*1024
            
        )
        self.netR_FC1 = nn.Sequential(
            # B*128*1024*1
            nn.Conv2d(128, opt.JOINT_NUM*4, kernel_size=(1, 1)),
            nn.BatchNorm2d(opt.JOINT_NUM*4),
            nn.ReLU(inplace=True),
            # B*4J*1024*1
		)
        
    def forward(self, x, y,x00):
		######################HPN-1##########################################################################################3
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        x10 = self.netR_1(x)
        # B*128*sample_num_level1*1
        x11 = torch.cat((y, x10),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        
        inputs_level2, inputs_level2_center = group_points_2(x11, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        
        # B*131*sample_num_level2*knn_K
        x20 = self.netR_2(inputs_level2)
        # B*256*sample_num_level2*1
        x21 = torch.cat((inputs_level2_center, x20),1)
        # B*259*sample_num_level2*1
        
        x30 = self.netR_3(x21)
        # B*1024*1*1
        x20=pointnetfc(inputs_level2_center,None,x20,x30)
        x20=self.netR_4(x20)
        x20=x20.unsqueeze(3)
        #B*256*sample_num_level2*1
        x10=pointnetfc(y,inputs_level2_center,x10,x20)
        x10=self.netR_5(x10)
        x10=x10.unsqueeze(3)
        #B*128*sample_num_level1*1
        x01=pointnetfc(x00[:,:3,:,:],y,None,x10)
        x01=self.netR_5(x01)
        x01=x01.unsqueeze(3)
        #B*128*1024*1
        heatmap_1 = netR_FC1(x01)
        ## B*4J*1024*1
        #################################HPN-2##########################################################################################3
        x=torch.cat((x00,heatmap_1,x01),1)
        x=group_points_3(x,opt)
        # x: B*(3+128+4J)*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        x10 = self.netR_11(x)
        # B*128*sample_num_level1*1
        x11 = torch.cat((y, x10),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        
        inputs_level2, inputs_level2_center = group_points_2(x11, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        
        # B*131*sample_num_level2*knn_K
        x20 = self.netR_2(inputs_level2)
        # B*256*sample_num_level2*1
        x21 = torch.cat((inputs_level2_center, x20),1)
        # B*259*sample_num_level2*1
        
        x30 = self.netR_3(x21)
        # B*1024*1*1
        x20=pointnetfc(inputs_level2_center,None,x20,x30)
        x20=self.netR_4(x20)
        x20=x20.unsqueeze(3)
        #B*256*sample_num_level2*1
        x10=pointnetfc(y,inputs_level2_center,x10,x20)
        x10=self.netR_5(x10)
        x10=x10.unsqueeze(3)
        #B*128*sample_num_level1*1
        x00=pointnetfc(x00,y,None,x10)
        x00=self.netR_5(x00)
        x00=x00.unsqueeze(3)
        #B*128*1024*1
        heatmap_2 = netR_FC1(x00)
        ## B*4J*1024*1
		
        return heatmap_1,heatmap_2
