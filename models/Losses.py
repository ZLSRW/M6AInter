import torch
import torch.nn as nn
import math
from .Utils import *
import numpy as np
# 高斯距离
from sklearn.metrics.pairwise import euclidean_distances


class Position_cluster1(nn.Module):
    def __init__(self, temperature):
        super(Position_cluster1, self).__init__()

        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def Position_mask(self,L,a,b):
        # 考虑拉普拉斯矩阵上的增益方向，即散度大小。散度为正，则越发散，为负则相对不那么发散。
        # 基于这个正负作为正负样本的选择标准，即负值为正样本，正值为负样本。
        mask1 = torch.ones((a,b))
        mask2 = torch.ones((a,b))
        mask1 = mask1.fill_diagonal_(0) #先对对角线赋0值
        mask2 = mask2.fill_diagonal_(0) #先对对角线赋0值
        pos_mask,neg_mask =pos_neg_mask(L,mask1,mask2)
        return pos_mask,neg_mask

    def forward(self,Position_cluster, L):
        a,b = L.size()
        Gaussian_matrix1 = torch.tensor(euclidean_distances(Position_cluster.detach().numpy()))
        Gaussian_matrix = Gaussian_matrix1 / self.temperature

        pos_mask,neg_mask = self.Position_mask(L,a,b)

        p_i = Position_cluster.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()


        pos_sim=((pos_mask*Gaussian_matrix)*(L)).detach().numpy()
        neg_sim=(neg_mask*Gaussian_matrix).detach().numpy()
        pos_sim_value=-np.sum(pos_sim)*10
        neg_sim_value=np.sum(neg_sim)

        loss=-math.log(pos_sim_value/neg_sim_value)

        return loss+ne_i

class Representation_Loss(nn.Module):

    def __init__(self, temperature):
        super(Representation_Loss, self).__init__()

        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, z_i.size(0))
        sim_j_i = torch.diag(sim, z_i.size(0))

        mask=self.mask_correlated_samples(z_i.size(0))

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        bias = self.criterion(logits, labels)
        bias /= N

        return bias
