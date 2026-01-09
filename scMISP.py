from __future__ import print_function, division
import warnings
from opt import args
from encoder import *

warnings.filterwarnings("ignore")

class scMISP(nn.Module):

    def __init__(self, ae1, ae2, agcn1, agcn2):
        super(scMISP, self).__init__()

        self.ae1 = ae1
        self.ae2 = ae2
        self.agcn1 = agcn1
        self.agcn2 = agcn2

        # Subspace bases proxy
        self.D1 = Parameter(torch.Tensor(args.n_z, args.n_clusters))
        self.D2 = Parameter(torch.Tensor(args.n_z, args.n_clusters))

    def forward(self, x1, adj1, x2, adj2, pretrain=False):

        h1_r, h2_r, z1 = self.ae1.encoder(x1)
        h1_a, h2_a, z2 = self.ae2.encoder(x2)

        x_hat1 = self.ae1.decoder(z1)
        x_hat2 = self.ae2.decoder(z2)

        if not pretrain:
            d = args.d
            s = None
            eta = args.eta

            # Calculate subspace affinity of D and Z1
            for i in range(args.n_clusters):
                si = torch.sum(torch.pow(torch.mm(z1, self.D1[:, i * d:(i + 1) * d]), 2), 1, keepdim=True)
                if s is None:
                    s = si
                else:
                    s = torch.cat((s, si), 1)
            s = (s + eta * d) / ((eta + 1) * d)
            s1 = (s.t() / torch.sum(s, 1)).t()

            s = None
            # Calculate subspace affinity of D and Z2
            for i in range(args.n_clusters):
                si = torch.sum(torch.pow(torch.mm(z2, self.D2[:, i * d:(i + 1) * d]), 2), 1, keepdim=True)
                if s is None:
                    s = si
                else:
                    s = torch.cat((s, si), 1)
            s = (s + eta * d) / ((eta + 1) * d)
            s2 = (s.t() / torch.sum(s, 1)).t()

            z1_agcn, p1 = self.agcn1(x1, h1_r, h2_r, z1, adj1)
            z2_agcn, p2 = self.agcn2(x2, h1_a, h2_a, z2, adj2)
        else:
            z1_agcn, z2_agcn = None, None
            s1, s2 = None, None
            p1, p2 = None, None

        return x_hat1, x_hat2, s1, s2, p1, p2, z1, z2, z1_agcn, z2_agcn




