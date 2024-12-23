import torch
import torch.nn as nn
import math
import numpy as np

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size#256
        mask = torch.ones((N, N))#[256,256]
        mask = mask.fill_diagonal_(0)#[256,256]，对角线是0
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0#[256,256]正样本处均为0，即右上角和左下角[128,128]的对角线
        mask = mask.bool()#[256,256]转化为bool值
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size#256
        z = torch.cat((z_i, z_j), dim=0)#[256,128]
        sim = torch.matmul(z, z.T) / self.temperature#相似度矩阵
        sim_i_j = torch.diag(sim, self.batch_size)#128,右上角的对角线
        sim_j_i = torch.diag(sim, -self.batch_size)#128,左下角的对角线
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)#[256,1]正样本拼接后转置
        negative_samples = sim[self.mask].reshape(N, -1)#去除所有的正样本后reshape，[256,254]
        labels = torch.zeros(N).to(positive_samples.device).long()#全零标签正样本
        logits = torch.cat((positive_samples, negative_samples), dim=1)#拼接正负样本，每行代表第几个正样本和所有负样本的相似度
        loss = self.criterion(logits, labels)#损失
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)#20，按行相加
        p_i /= p_i.sum()#归一化
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()#Negative Entropy
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()#转置
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)#[40,128]

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)      
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + ne_loss
