import torch.nn as nn
import torch
from torch.nn.functional import normalize
from copy import deepcopy
import torchvision
from PIL import ImageFilter, ImageOps
import random
import pandas as pd

class Network_VTCC(nn.Module):
    def __init__(self, vtcc, feature_dim, class_num):
        super(Network_VTCC, self).__init__()
        self.vtcc = vtcc
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.mid_dim = self.vtcc.dim * 4

        self.instance_projector = nn.Sequential(
            nn.Linear(self.vtcc.dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.vtcc.dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j, x_ii):
        x_ii = x_ii.unsqueeze(1)
        shape = x_i.shape[0]
        x_i = self.vtcc.patch_embed(x_i)
        cls_token = self.vtcc.cls_token.expand(x_i.shape[0], -1, -1)
        x_i = torch.cat((cls_token, x_i), dim=1)
        x_i = self.vtcc.pos_drop(x_i + self.vtcc.pos_embed)#[128,197,384]
        x_i = torch.cat((x_ii, x_i), dim=1)
        x_j = self.vtcc.patch_embed(x_j)
        cls_token = self.vtcc.cls_token.expand(x_j.shape[0], -1, -1)
        x_j = torch.cat((cls_token, x_j), dim=1)
        x_j = self.vtcc.pos_drop(x_j + self.vtcc.pos_embed)#[128,197,384]
        x_j = torch.cat((x_ii, x_j), dim=1)
        x_i = self.vtcc.blocks[0](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[0](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[1](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[2](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[2](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[3](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[4](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[4](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[5](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[6](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[6](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[7](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.norm(x_i)#[128,197,384]
        x_j = self.vtcc.norm(x_j)#[128,197,384]
        h_i = self.vtcc.pre_logits(x_i[:, 0])
        h_j = self.vtcc.pre_logits(x_j[:, 0])
        z_i = normalize(self.instance_projector(h_i), dim=1)#[128, 128]
        z_j = normalize(self.instance_projector(h_j), dim=1)
        c_i = self.cluster_projector(h_i)#[128, 20]
        c_j = self.cluster_projector(h_j)
        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x, x_ii):
        x_ii = x_ii.unsqueeze(1)
        x_i = x
        x_j = x
        shape = x.shape[0]
        x_i = self.vtcc.patch_embed(x_i)
        cls_token = self.vtcc.cls_token.expand(x_i.shape[0], -1, -1)
        x_i = torch.cat((cls_token, x_i), dim=1)
        x_i = self.vtcc.pos_drop(x_i + self.vtcc.pos_embed)#[128,197,384]
        x_i = torch.cat((x_ii, x_i), dim=1)
        x_j = self.vtcc.patch_embed(x_j)
        cls_token = self.vtcc.cls_token.expand(x_j.shape[0], -1, -1)
        x_j = torch.cat((cls_token, x_j), dim=1)
        x_j = self.vtcc.pos_drop(x_j + self.vtcc.pos_embed)#[128,197,384]
        x_j = torch.cat((x_ii, x_j), dim=1)
        x_i = self.vtcc.blocks[0](x_i)
        x_j = self.vtcc.blocks[0](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[1](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[2](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[2](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[3](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[4](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[4](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[5](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[6](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[6](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[7](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.norm(x_i)#[128,197,384]
        x_j = self.vtcc.norm(x_j)#[128,197,384]
        h_i = self.vtcc.pre_logits(x_i[:, 0])
        h_j = self.vtcc.pre_logits(x_j[:, 0])
        z_i = normalize(self.instance_projector(h_i), dim=1)#[128, 128]
        z_j = normalize(self.instance_projector(h_j), dim=1)
        c_i = self.cluster_projector(h_i)#[128, 20]
        c_j = self.cluster_projector(h_j)
        c = torch.argmax(c_j, dim=1)
        return c
    
    def forward_vs(self, x, x_ii):
        x_ii = x_ii.unsqueeze(1)
        x_i = x
        x_j = x
        shape = x.shape[0]
        x_i = self.vtcc.patch_embed(x_i)
        cls_token = self.vtcc.cls_token.expand(x_i.shape[0], -1, -1)
        x_i = torch.cat((cls_token, x_i), dim=1)
        x_i = self.vtcc.pos_drop(x_i + self.vtcc.pos_embed)#[128,197,384]
        x_i = torch.cat((x_ii, x_i), dim=1)
        x_j = self.vtcc.patch_embed(x_j)
        cls_token = self.vtcc.cls_token.expand(x_j.shape[0], -1, -1)
        x_j = torch.cat((cls_token, x_j), dim=1)
        x_j = self.vtcc.pos_drop(x_j + self.vtcc.pos_embed)#[128,197,384]
        x_j = torch.cat((x_ii, x_j), dim=1)
        x_i = self.vtcc.blocks[0](x_i)
        x_j = self.vtcc.blocks[0](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[1](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[2](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[2](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[3](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        x_i = self.vtcc.blocks[4](x_i)#[128, 197, 384]
        x_j = self.vtcc.blocks[4](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[5](X)#[128, 197, 384]
        x_i = X[:shape,:,:]
        x_j = X[shape:,:,:]
        att = self.vtcc.blocks[6](x_i, return_attention=True)#[128, 197, 384]
        x_j = self.vtcc.blocks[6](x_j)#[128, 197, 384]
        X = torch.cat([x_i, x_j], dim = 0)
        X = self.vtcc.blocks[7](X)#[128, 197, 384]
        return att