from pointnet import PointNet
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):#返回点与点之间的距离，还没开方
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
		idx表示是在每个Batch中提取的点的索引，他可以是一个数也可以是一个列表
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
		注意这里返回的点也就是new_point是4个维度的。
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

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
		返回的是索引，表示在每个batch中找出npoint个远点
    """
    device = xyz.device
    B, N, C = xyz.shape
    S = npoint
    centroids = torch.zeros(B, S, dtype=torch.long).to(device)#质心初始化为0，
    distance = torch.ones(B, N).to(device) * 1e10
	distance = distance.to(torch.long)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#[1*B]大小 随机初始化第一个点
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]#这一步操作很巧妙
		'''
		从剩余点中，取一个距离点集A最远的点，一直采样到目标数量N为止。
		一个点P到到点集A距离的定义：
		P点到A中距离最近的一个点的距离，min(dis(P,A1),...dis(P,An))。
		'''
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
		返回的是对于每个batch，每一个S的nsample个radius范围内的点的。。。索引
		输入一个半径，以及采样点的个数
		如果出现了radius内没有足够的nsample个点（哪怕有一个满足的点都行，因为代码会
		复制最近的那个点，然后补充到nsample。但是如果一个点都没有，那么在后面的
		index_points(...)中会出错。
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    K = nsample
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])#[B,S,N]
    sqrdists = square_distance(new_xyz, xyz)#[B,S,N]
    group_idx[sqrdists > radius**2] = N#对于每一个B，将sqrdists>radius**2的部分设置为N，其他的数值都小于N
	group_idx = group_idx.sort(dim=-1)[0][:,:,:K]#sort函数取[0]表示value，然后取每batch每行的前K个
    group_first = group_idx[:,:,0].view(B, S, 1).repeat([1, 1, K])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, C] 采样后的点
        new_points: sampled points data, [B, npoint, nsample, C+D]
		返回每一堆“圆内点”的局部坐标。
    """
    B, N, C = xyz.shape
    S = npoint

    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)#按球和半径取点
    grouped_xyz -= new_xyz.view(B, S, 1, C)#分别减去质心
    if points is not None:
        grouped_points = index_points(points, idx)#把对应的特征提取出来。
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)#把特征接到后面
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]  
		这里应该是每个batch假设采样一个点，然后该点采取圆内点的时候，提取的是所有点的特征，也就是全局特征
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points#后面这个new_xyz也没用上

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
    
    def forward(self, xyz, points):
        """
        Input: 
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S] 这个是提取的质心
            new_points_concat: sample points feature data, [B, D', S] 这个D' 表示的是mlp的最后一项
			注意返回值，点数是一样的，只是每个数据的纬度不同
        """
        xyz = xyz.permute(0, 2, 1)#更换维度
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
		
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
		#new_points 的shape [B, npoint, nsample, C+D] max处理后，变成了 [B, C+D, npoint]
		#max 这里的意义，把 nsample 这个维度去掉了。nsample 代表了在一个质心取多少个球内点
		#max 后，把对于一个质心的nsample个点中最大的那个取出来了，相当于提取了局部特征
        new_xyz = new_xyz.permute(0, 2, 1)
		#new_xyz 的shape [B, C, npoint]
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
    
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
			与Ssg类似，不同的是Msg提取了不同半径的圆内的点，半径小的，采样的点少一些，做MLP时提取的特征的深度就浅一些。
			最后得到的每个点的特征的维度会比Ssg的大很多
		"""
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1) #[B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0] #[B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat
		
		
#self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
#l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
			实际上，xyz2是xyz1采样后的点，首先用距离计算一个权重，距离小的权重大，然后归一化权重。
			
			原文：
			We achieve feature propagation by interpolating feature values
			f of Nl points at coordinates of the NL-1 points. Among the many choices for interpolation, we
			use inverse distance weighted average based on k nearest neighbors (as in Eq. 2, in default we use
			p = 2, k = 3). The interpolated features on NL-1 points are then concatenated with skip linked point
			features from the set abstraction level.
			重点是 interpolated_points 的意义：首先，对于NL的每一个点，计算出NL-1层上离该点的最近的3个点，然后将这三个点的
			feature提取出来，按权重，最后提取成一个feature，这个feature的前俩个维度和points1的维度相同，然后进行cat。
			如果没有 points1 的话，就不用cat。
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)#如果S==1，情况就是 xyz2 是最后一层的点，需要把数据复制n次。
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)#排序的是xyz1的某一个点到xyz2的点的距离。
            dists, idx = dists[:,:,:3], idx[:,:,:3] #[B, N, 3]  取前三个近的点
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
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNet2ClsMsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsMsg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1,0.2,0.4], [16,32,128], 0, [[32,32,64], [64,64,128], [64,96,128]])
		self.sa2 = PointNetSetAbstractionMsg(128, [0.2,0.4,0.8], [32,64,128], 320, [[64,64,128], [128,128,256], [128,128,256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

class PointNet2ClsSsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsSsg, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128,128,256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 40)#最后分类是40个
    
    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
		#注意两个return的意义，l1_xyz是提取的质心的集合，l1_points是提取的每个质心代表的球内的局部特征，这两个值表示点的数量是一样的，但是维度上不同。一个是坐标，一个是特征
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
		#在最后一个setAbstraction 中 group_all 是True，这里假设npoint是1，那么l3_points的维度是[B, 1, D']这里D'是1024
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

class PointNet2PartSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2PartSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 64, 3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 256, 1024], True)
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)#做partseg 就要用conv1d。
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x

"""
    Custom segmentation network for tensorbody dataset
"""
class PointNet2Seg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2Seg, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 64, 3, [64, 128, 256], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 256 + 3, [256, 512, 1024], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 1024 + 3, [1024, 2048, 4096], True)
        self.fp3 = PointNetFeaturePropagation(5120, [1024, 1024])
        self.fp2 = PointNetFeaturePropagation(1280, [1024, 1024])
        self.fp1 = PointNetFeaturePropagation(1024, [1024, 1024])
        self.conv1 = nn.Conv1d(1024, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(1024, num_classes, 1)

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(name)
                continue
            own_state[name].copy_(param)

if __name__ == '__main__':
    # for i in range(10):
    #     t = time()
    #     xyz = torch.rand(16, 3, 2048).cuda()
    #     net = PointNet2SemSeg(2048)
    #     net.cuda()
    #     x = net(xyz)
    #     timeit('it', t)

    # xyz1 = torch.rand(4, 3, 2048).cuda()
    # xyz2 = torch.rand(4, 3, 512).cuda()
    # points2 = torch.rand(4, 64, 2048).cuda()
    # net = PointNetFeaturePropagation(64, [128, 256])
    # net.cuda()
    # new_points = net(xyz1, xyz2, None, points2)
    # print(new_points.shape)

    # xyz = torch.rand(8, 3, 2048).cuda()
    # net = PointNet2SemSeg(2048)
    # net.cuda()
    # x = net(xyz)
    # print(x.shape)

    a = torch.load('part_seg/8.pth')
    for name ,param in a.items():
        print(name)