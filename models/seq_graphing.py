import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
from .tcn import *
# from .Causal_Dilate_Network import *
import pandas as pd
import csv

from .Losses import *
from .CovLSTM import *
from .fuse import *
from .SAE import *



# 实例权重确定并计算位置图
class InstanceWeight(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(InstanceWeight, self).__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize

        self.L1 = nn.Sequential(
            nn.Linear(self.inputsize, self.outputsize),
            # nn.ReLU()
            nn.Tanh()
            # nn.LeakyReLU()
        )

        self.L2 = nn.Sequential(
            nn.Linear(self.inputsize, self.outputsize),
            # nn.Tanh()
            # nn.ReLU()
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.inputsize, self.outputsize)  # 注意力权重
        return

    def forward(self, All_attention, x_onehot):  # s输入为一个bx41x41的图所对应的邻接矩阵

        l1 = self.L1(All_attention)
        l2 = self.L2(All_attention)
        W = self.attention_weights(l1 * l2)  # 共享的权重矩阵（不一定非要说明共享）41x41xK
        W_i = F.softmax(W, dim=-1)

        All_attention = W_i * All_attention
        Position_attention = torch.sum(All_attention, dim=-1)

        W_t = torch.sum(W, dim=1).unsqueeze(1)

        x_onehot = x_onehot.squeeze()
        x_onehot = x_onehot.reshape(-1, 41, 4).permute(1, 2, 0)  # 41x4xk
        x_type = W_t * x_onehot
        x_type = torch.sum(x_type, dim=-1)
        x_type = torch.nn.functional.normalize(x_type, p=2, dim=1)  # 每个元素均为0到1之间的概率
        x_type = torch.softmax(x_type, dim=1)  # 每一行的和为1

        Position_attention = max_min_2D(Position_attention)  # 看看效果，是否需要执行标准化
        Position_attention = 0.5 * (Position_attention + Position_attention.T)  # 将位置图转成对称

        return Position_attention, x_type


class Representation_discrepancy(nn.Module):
    def __init__(self):
        super(Representation_discrepancy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.Representation_Loss = Representation_Loss(0.5)

    def forward(self, R1, R2):  # b*256

        coff = 1
        bias = self.Representation_Loss(R1, R2)
        bias = coff / bias

        return bias


class Gen_Constractive_Clustering(nn.Module):
    def __init__(self, unit, cluster_num):
        super(Gen_Constractive_Clustering, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.Position_cluster1 = Position_cluster1(0.5)
        self.unit = unit
        self.cluster_num = cluster_num

        self.cluster_projector = nn.Sequential(
            nn.Linear(self.unit, self.unit),
            nn.ReLU(),
            nn.Linear(self.unit, self.cluster_num),
            nn.Softmax(dim=1))

    def forward(self, Position_attention, mul_L, All_attention):  #

        self.fc = nn.Sequential(
            nn.Linear(All_attention.size(0), All_attention.size(-1)),
            nn.Tanh(),
            nn.Linear(All_attention.size(-1), 64),
        )

        Position_cluster = self.cluster_projector(Position_attention)
        CLoss = self.Position_cluster1(Position_cluster, mul_L)
        motifs = torch.argmax(Position_cluster, dim=1)  # 返回各个位置所属的类别，（0到41）
        return motifs, CLoss


class multiDomainSeqLayer0(nn.Module):  # 该模块需要实现的职能是将拉普拉斯矩阵与原始特征相乘，并得到最终的序列表示。
    def __init__(self, ):
        super(multiDomainSeqLayer0, self).__init__()
        return

    def forward(self, x, mul_L):
        spectralSeq = torch.matmul(mul_L, x.permute(0, 2, 1))
        spectralSeq = self.relu(spectralSeq)

        return spectralSeq


class multiDomainSeqLayer1(nn.Module):  # 该模块需要实现的职能是将拉普拉斯矩阵与原始特征相乘，并得到最终的序列表示。
    def __init__(self, ):
        super(multiDomainSeqLayer1, self).__init__()
        self.relu=nn.ReLU()
        return

    def forward(self, x, mul_L):
        spectralSeq = torch.matmul(mul_L, x)
        spectralSeq =self.relu(spectralSeq)

        return spectralSeq


class Model(nn.Module):
    def __init__(self, batchsize, cluster_num, units, multi_layer, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.covLSTM = ConvNet_BiLSTM(41, 41, 1, 1, 0, 0.2)
        self.unit = units
        self.cluster_num = cluster_num
        self.alpha = leaky_rate
        self.batchsize = batchsize
        self.weight_graph = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.GRU = nn.GRU(self.unit, self.unit)
        self.bn = nn.BatchNorm1d(self.unit)

        self.multi_layer = multi_layer
        self.k = 1
        self.gcc = Gen_Constractive_Clustering(self.unit, self.cluster_num)
        self.rd = Representation_discrepancy()

        self.seqGraphBlock = nn.ModuleList()
        self.seqGraphBlock.extend(
            [multiDomainSeqLayer0()])

        self.seqGraphBlock.extend(
            [multiDomainSeqLayer1()])

        self.fc_shape = nn.Sequential(
            nn.Linear(self.unit, self.unit),
            nn.Tanh(),
            nn.Sigmoid(),
            nn.Linear(self.unit, self.unit),
        )

        self.fc_shape1 = nn.Sequential(
            nn.Linear(int(self.unit), 256),
            # nn.LeakyReLU(),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.fc_Kmer = nn.Sequential(
            nn.Linear(64, 256),
            # nn.LeakyReLU(),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.fc_shapex = nn.Sequential(

            nn.Linear(int(self.unit), int(self.unit)),
            # nn.LeakyReLU(),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(int(self.unit), 1),
        )

        self.fc_prob = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )


        self.relu = nn.LeakyReLU(self.alpha)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

        self.AE1 = AutoEncoder(256, 128)
        self.AE2 = AutoEncoder(128, 256)

    def cheb_polynomial(self, laplacian):  # 返回多阶拉普拉斯矩阵,这里使用的切比雪夫不等式的四阶式子

        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def cheb_polynomial_multi(self, laplacian):  # 返回多阶拉普拉斯矩阵,这里使用的切比雪夫不等式的四阶式子

        bat, N, N = laplacian.size()
        laplacian = laplacian.unsqueeze(1)
        first_laplacian = torch.zeros([bat, 1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=1)

        return multi_order_laplacian  # 32x12x4x100x100

    def seq_graph_ing(self, x, input_prob):

        input = x.permute(0, 2, 1)
        input = input.repeat(1, 1, self.unit)
        input, _ = self.GRU(input)
        attention = self.district_graph_attention(input, input_prob)
        attention_all = attention

        degree_all = torch.sum(attention, dim=2)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 0.1))

        laplacians = torch.matmul(diagonal_degree_hat,
                                  torch.matmul(degree_l - attention, diagonal_degree_hat))  # 得到拉普拉斯矩阵，类似GCN

        mul_L = self.cheb_polynomial(laplacians)
        return mul_L, attention_all, degree_all

    def seq_graph_ing_position(self, x, x_onehot, input_prob, bat):
        self.positioncode = TemporalConvNet(1, 1, bat)
        input = self.positioncode(x.contiguous())
        input = input.repeat(1, 1, self.unit)

        input, _ = self.GRU(input)
        attention = self.district_graph_attention(input, input_prob)

        hard_attention = max_min(attention)

        position_hard_attention = hard_attention.permute(1, 2, 0)

        self.position_generating = InstanceWeight(position_hard_attention.size(-1), position_hard_attention.size(-1))

        Position_attention, x_type = self.position_generating(position_hard_attention, x_onehot)

        return hard_attention, Position_attention, x_type

    def single_laplacian(self, Position_attention):

        hard_attention = Position_attention
        degree = torch.sum(hard_attention, dim=-1)

        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-6))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - hard_attention, diagonal_degree_hat))
        return laplacian

    def seq_graph_ing_Instance(self, x, input_prob, bat):
        self.positioncode = TemporalConvNet(1, 1, bat)
        input = self.positioncode(x.contiguous())
        input = input.repeat(1, 1, self.unit)


        input, _ = self.GRU(input)

        attention = self.district_graph_attention(input, input_prob)  # 32x100x100/32x12x512x512

        hard_attention = max_min(attention)
        attention = hard_attention

        position_attention = torch.mean(attention, dim=0)

        bat_attention = torch.matmul(torch.mean(attention, dim=1), torch.mean(attention, dim=1).T)


        degree = torch.sum(hard_attention, dim=-1)
        bat_degree = torch.sum(bat_attention, dim=-1)
        position_degree = torch.sum(position_attention, dim=1)

        hard_attention = 0.5 * (hard_attention + hard_attention.permute(0, 2, 1))
        bat_attention = 0.5 * (bat_attention + bat_attention.permute(1, 0))
        position_attention = 0.5 * (position_attention + position_attention.permute(0, 1))

        degree_l = tensor_diag(degree)
        diagonal_degree_hat = tensor_diag(1 / (torch.sqrt(degree) + 1e-6))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - hard_attention, diagonal_degree_hat))

        bat_degree_l = torch.diag(bat_degree)
        bat_diagonal_degree_hat = torch.diag(1 / (torch.sqrt(bat_degree) + 0.1))
        bat_laplacians = torch.matmul(bat_diagonal_degree_hat,
                                      torch.matmul(bat_degree_l - bat_attention,
                                                   bat_diagonal_degree_hat))

        position_degree_l = torch.diag(position_degree)
        position_diagonal_degree_hat = torch.diag(1 / (torch.sqrt(position_degree) + 0.1))
        position_laplacians = torch.matmul(position_diagonal_degree_hat,
                                           torch.matmul(position_degree_l - position_attention,
                                                        position_diagonal_degree_hat))

        mul_L = self.cheb_polynomial_multi(laplacian)
        bat_mul_L = self.cheb_polynomial(bat_laplacians)
        position_mul_L = self.cheb_polynomial(position_laplacians)

        return mul_L, attention, bat_mul_L, bat_attention, position_mul_L, position_attention  # 返回一个多阶的拉普拉斯矩阵，以及一个注意力矩阵（均为32x12x512x512的矩阵）

    def district_graph_attention(self, input, input_prob):
        bat, N, fea = input.size()  # 32 140 140
        key = torch.matmul(input, self.weight_key)  # 32x100x1
        query = torch.matmul(input, self.weight_query)  # 32x100x1

        data = query * key.permute(0, 2, 1)
        data = self.bn(data)
        data = self.relu(data)
        attention = F.softmax(data, dim=2)  # attention的值在0.02左右
        attention = attention * (1 + input_prob)

        return attention

    def motif_adjust(self, motifs, x_type, All_attention):


        motif_mask = torch.eq(motifs.unsqueeze(1), motifs.unsqueeze(0)).int()  # 相同为1不同为0
        type_mask = torch.matmul(x_type, x_type.T)
        mt_mask = motif_mask * type_mask

        All_attention = All_attention * mt_mask

        return All_attention


    def corre_attention_computing(self, All_attention):

        self.fc_shape0 = nn.Sequential(
            nn.Linear(All_attention.size(0), All_attention.size(0)),
            nn.ReLU(),
        )

        self.fc_final = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
        )

        All_attention = self.fc_shapex(All_attention.float())
        All_attention = All_attention.squeeze()
        corre_attention = torch.matmul(All_attention, All_attention.T)
        corre_attention = self.fc_shape0(corre_attention)

        corre_attention = max_min_2D(corre_attention)
        corre_attention = 0.5 * (corre_attention + corre_attention.T)

        return corre_attention

    def forward(self, x, input_prob, x_kmer, x_onehot):

        All_attention, Position_attention, x_type = self.seq_graph_ing_position(x, x_onehot, input_prob, x.size(
            0))
        Position_mul = self.single_laplacian(Position_attention)

        motifs, CLoss = self.gcc(Position_attention, Position_mul, All_attention)


        All_attention = self.motif_adjust(motifs, x_type, All_attention)
        Bat_attention = self.corre_attention_computing(All_attention)

        Bat_mul = self.single_laplacian(Bat_attention)
        Bat_mul = self.cheb_polynomial(Bat_mul)

        X = self.fc_shape1(x).squeeze()

        Bat_mul = torch.sum(Bat_mul, dim=0)

        result = []

        for block in range(2):
            if block == 1:
                forecast = self.seqGraphBlock[block](X, Bat_mul)
                result.append(forecast)

        forecast = X + result[0]

        x_kmer = self.fc_Kmer(x_kmer.squeeze())

        Bias = self.rd(forecast, x_kmer)
        forecast = (forecast + Bias) * x_kmer

        AEList = [self.AE1, self.AE2, ]
        self.SAE = SAE(AEList)
        forecast = self.SAE(forecast)

        forecast_prob = self.fc_prob(forecast)

        return forecast, forecast_prob, CLoss, motifs, x_type
