import argparse
import torch

parser = argparse.ArgumentParser(
    description='EDESC training',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=float, default=1600)
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--eta', default=5, type=int)
parser.add_argument('--method', type=str, default='euc')
parser.add_argument('--update_interval', default=20, type=int)
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--first_view', type=str, default='RNA')

# parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset', type=str, default='pbmccite')
parser.add_argument('--pretrain_path', type=str, default='data/pbmc10k')
parser.add_argument('--alpha_value', type=float, default=0.1)

# alpha & beta & lambda need to be modified according to different datasets.
parser.add_argument('--alpha', default=0.01, type=float, help='coefficient of subspace bases constraint')
parser.add_argument('--beta', default=5, type=float, help='coefficient of KL loss')
parser.add_argument('--lambda1', default=0.01, type=float)

parser.add_argument('--k', default=10, type=int)
parser.add_argument('--gamma', default=5, type=float)

parser.add_argument('--ae_n_enc_1', type=int, default=256)
parser.add_argument('--ae_n_enc_2', type=int, default=128)
parser.add_argument('--ae_n_dec_1', type=int, default=128)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--n_d1', type=int, default=100)  #视角1 输入维度
parser.add_argument('--n_d2', type=int, default=14)  #视角2 输入维度
parser.add_argument('--n_z', type=int, default=20)   # n_cluster * d
parser.add_argument('--n_clusters', default=4, type=int)
parser.add_argument('--d', default=6, type=int)

parser.add_argument('--nmi', default=0, type=int)

args = parser.parse_args()