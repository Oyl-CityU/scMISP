from __future__ import division, print_function
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from munkres import Munkres
import torchvision
from torchvision import datasets, transforms
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import adjusted_mutual_info_score
import h5py
import scipy.io
from sklearn import preprocessing
import math
from opt import args
import os
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

def refined_subspace_affinity(s):
    weight = s ** 2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def numpy_to_torch(a, sparse=False):
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a

def distribution_loss(Q, P):
    loss = F.kl_div((Q.log()), P)
    # loss = F.kl_div(Q[0].log(), P, reduction='batchmean')
    return loss

def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def get_adj(count, k=15, mode="connectivity"):
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n

def load_graph(fea, view, k, label):
    folder ='./data/' + args.dataset + '/'
    graph_path = '{}{}_{}.npz'.format(folder, view, k)
    if not os.path.exists(graph_path):
        _, adj = get_adj(count=fea, k=k)
        num = len(label)
        counter = 0
        for i in range(adj.shape[0]):
            for j in range(adj.shape[0]):
                if adj[i][j] == 0 or i==j:
                    pass
                else:
                    if label[i] != label[j]:
                        counter += 1
        print('error rate: {}'.format(counter / (num * k)))
        sp.save_npz(graph_path, sp.csr_matrix(adj))

    adj = sp.load_npz(graph_path).toarray()

    return adj

def load_dataset(data_path='./data/pbmccite', omic_2='ADT'):
    import os
    if not os.path.exists(os.path.join(data_path, 'RNA_fea.npy')) and os.path.exists(os.path.join(data_path, '{}_fea.npy'.format(omic_2)))\
            and os.path.exists(os.path.join(data_path, 'label.npy')):
        assert False,"Incomplete files"
    omic1_fea = np.load(os.path.join(data_path, 'RNA_fea.npy'),allow_pickle=True)
    omic2_fea = np.load(os.path.join(data_path, '{}_fea.npy'.format(omic_2)),allow_pickle=True)
    label = np.load(os.path.join(data_path, 'label.npy'),allow_pickle=True)
    A1 = load_graph(omic1_fea, 'RNA', args.k, label)
    A2 = load_graph(omic2_fea, omic_2, args.k, label)
    x1 = omic1_fea
    x2 = omic2_fea
    y = label
    # has been shuffled
    print(('omic1 samples', x1.shape))
    print(('omic2 samples', x2.shape))
    return x1, A1, x2, A2, y

def LoadDatasetByName(dataset_name, omic2):
    data_path = './data/' + dataset_name
    x1, A1, x2, A2, y = load_dataset(data_path, omic2)

    return x1, A1, x2, A2, y

class LoadDataset(Dataset):

    def __init__(self, dataset_name, omic2):
        self.x1, self.A1, self.x2, self.A2, self.y = LoadDatasetByName(dataset_name, omic2)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(np.array(self.x1[idx])), torch.from_numpy(np.array(self.A1[idx])),
                torch.from_numpy(np.array(self.x2[idx])), torch.from_numpy(np.array(self.A2[idx])),
                torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)))

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        print(num_class1)
        print(numclass2)
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)

    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1 = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1


def eva(y_true, y_pred, show_details=True):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    if show_details:
        print("\n", "ARI: {:.4f},".format(ari), "NMI: {:.4f},".format(nmi), "AMI: {:.4f}".format(ami),
              "ACC: {:.4f},".format(acc))

    return ari, nmi, ami, acc, f1

def assignment(Q, y):
    y_pred = torch.argmax(Q, dim=1).data.cpu().numpy()
    ari, nmi, ami, acc, f1 = eva(y, y_pred, show_details=False)
    return ari, nmi, ami, acc, f1, y_pred
