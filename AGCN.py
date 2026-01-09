import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from encoder import *
from opt import args

class DIF_MSIF_GCN(nn.Module):
    def __init__(self, n_d):
        super(DIF_MSIF_GCN, self).__init__()

        self.agcn_0 = GNNLayer(n_d, args.ae_n_enc_1)
        self.agcn_1 = GNNLayer(args.ae_n_enc_1, args.ae_n_enc_2)
        self.agcn_2 = GNNLayer(args.ae_n_enc_2, args.n_z)
        self.agcn_z = GNNLayer(args.ae_n_enc_1 + args.ae_n_enc_2 + 2 * args.n_z, args.n_clusters)

        self.mlp = MLP_L(args.ae_n_enc_1 + args.ae_n_enc_2 + 2*args.n_z)
        self.mlp_1 = MLP_1(2 * args.ae_n_enc_1)
        self.mlp_2 = MLP_2(2 * args.ae_n_enc_2)

    def forward(self, x, h1, h2, z, adj):
        # DIF-GCN
        z1 = self.agcn_0(x, adj) # z1

        m1 = self.mlp_1(torch.cat((z1, h1), 1) )
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:,0], [args.num_sample, 1])
        m12 = torch.reshape(m1[:,1], [args.num_sample, 1])
        m11_broadcast = m11.repeat(1, args.ae_n_enc_1)
        m12_broadcast = m12.repeat(1, args.ae_n_enc_1)
        z2 = self.agcn_1(m11_broadcast.mul(z1)+m12_broadcast.mul(h1), adj) # z2

        # z3
        m2 = self.mlp_2(torch.cat((z2, h2),1) )
        m2 = F.normalize(m2, p=2)
        m21 = torch.reshape(m2[:,0], [args.num_sample, 1])
        m22 = torch.reshape(m2[:,1], [args.num_sample, 1])
        m21_broadcast = m21.repeat(1,args.ae_n_enc_2)
        m22_broadcast = m22.repeat(1,args.ae_n_enc_2)
        z3 = self.agcn_2(m21_broadcast.mul(z2)+m22_broadcast.mul(h2), adj) # z3

        # MSIF-GCN
        u = self.mlp(torch.cat((z1, z2, z3, z), 1))
        u = F.normalize(u, p=2)
        u0 = torch.reshape(u[:, 0], [args.num_sample, 1])
        u1 = torch.reshape(u[:, 1], [args.num_sample, 1])
        u2 = torch.reshape(u[:, 2], [args.num_sample, 1])
        u3 = torch.reshape(u[:, 3], [args.num_sample, 1])

        tile_u0 = u0.repeat(1, args.ae_n_enc_1)
        tile_u1 = u1.repeat(1, args.ae_n_enc_2)
        tile_u2 = u2.repeat(1, args.n_z)
        tile_u3 = u3.repeat(1, args.n_z)

        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z)), dim=1)
        net_output = self.agcn_z(net_output, adj, active=False)
        predict = F.softmax(net_output, dim=1)

        return net_output, predict
