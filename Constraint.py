import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args


class D_constraint1(torch.nn.Module):

    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d, alpha):
        # I = torch.eye(d.shape[1]).cuda()
        I = torch.eye(d.shape[1], device=args.device)
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return 	alpha * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):

    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim,n_clusters, alpha):
        #S = torch.ones(d.shape[1],d.shape[1]).cuda()
        S = torch.ones(d.shape[1], d.shape[1], device=args.device)
        zero = torch.zeros(dim, dim)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return alpha * loss_d2_constraint


