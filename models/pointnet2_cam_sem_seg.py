import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)
    return idx

class ContextAggregationModule(nn.Module):
    def __init__(self, channels, k=16, groups=4):
        super(ContextAggregationModule, self).__init__()
        self.k = k
        self.channels = channels
        self.inter_channels = channels // 4

        self.query_conv = nn.Conv1d(channels, self.inter_channels, 1, groups=groups)
        self.key_conv = nn.Conv1d(channels, self.inter_channels, 1, groups=groups)
        self.value_conv = nn.Conv1d(channels, channels, 1, groups=groups)

        self.pos_proj = nn.Conv2d(3, self.inter_channels, 1)

        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x, pos):
        B, C, N = x.shape

        idx = knn_point(self.k, pos.permute(0, 2, 1), pos.permute(0, 2, 1))

        query = self.query_conv(x).permute(0, 2, 1)  
        key = self.key_conv(x)  
        value = self.value_conv(x)  

        idx_expand = idx.unsqueeze(1).expand(B, self.inter_channels, N, self.k)
        key_neighbors = torch.gather(key.unsqueeze(-1).expand(-1, -1, -1, self.k), 2, idx_expand)

        idx_expand_val = idx.unsqueeze(1).expand(B, C, N, self.k)
        value_neighbors = torch.gather(value.unsqueeze(-1).expand(-1, -1, -1, self.k), 2, idx_expand_val)

        pos_expand = pos.unsqueeze(-1).expand(B, 3, N, self.k)
        idx_expand_pos = idx.unsqueeze(1).expand(B, 3, N, self.k)
        pos_neighbors = torch.gather(pos.unsqueeze(-1).expand(-1, -1, -1, self.k), 2, idx_expand_pos)

        relative_pos = pos_expand - pos_neighbors
        pos_bias = self.pos_proj(relative_pos)

        query = query.permute(0, 2, 1).unsqueeze(-1)

        energy = (query * key_neighbors + pos_bias).sum(dim=1)
        attention = F.softmax(energy, dim=-1)

        out = (value_neighbors * attention.unsqueeze(1)).sum(dim=-1)
        out = self.bn(out + x)
        out = F.relu(out)
        return out


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)

        self.context_l2 = ContextAggregationModule(128, k=16, groups=4)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_xyz = xyz
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l2_points = self.context_l2(l2_points, l2_xyz)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        return F.nll_loss(pred, target, weight=weight)


if __name__ == '__main__':
    pass
