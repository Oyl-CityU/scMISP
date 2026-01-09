from __future__ import print_function, division
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from torch.optim import Adam
from encoder import *
from scMISP import scMISP
from AGCN import DIF_MSIF_GCN
from utils import *
import warnings
from InitializeD import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time
import tqdm
from opt import args


warnings.filterwarnings("ignore")

def pretrain(model, x1, A1, x2, A2):
    params = list(model.ae1.parameters()) + list(model.ae2.parameters())
    optimizer = Adam(params, lr=args.lr)
    pbar = tqdm.tqdm(range(200))
    for epoch in pbar:
        optimizer.zero_grad()
        x_hat1, x_hat2, _, _, _, _, z1, z2, _, _ = model(x1, A1, x2, A2, pretrain=True)
        loss_re = F.mse_loss(x_hat1, x1) + F.mse_loss(x_hat2, x2)
        x1_norm = F.normalize(x1, dim=1, p=2)
        x2_norm = F.normalize(x2, dim=1, p=2)
        z1_norm = F.normalize(z1, dim=1, p=2)
        z2_norm = F.normalize(z2, dim=1, p=2)
        x1_sim = x1_norm @ x1_norm.T
        x2_sim = x2_norm @ x2_norm.T
        z1_sim = z1_norm @ z1_norm.T
        z2_sim = z2_norm @ z2_norm.T
        loss_sim1 = F.mse_loss(x1_sim, z1_sim)
        loss_sim2 = F.mse_loss(x2_sim, z2_sim)
        loss_sim = loss_sim1 + loss_sim2
        loss = loss_re + args.lambda1 * loss_sim
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(loss.item())})
    pbar.close()

def train(model, x1, A1, x2, A2, y):
    start = time.time()

    pretrain(model, x1, A1, x2, A2)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Cluster parameter initiate
    with torch.no_grad():
        _, _, _, _, _, _, z1, z2, _, _ = model(x1, A1, x2, A2)

    # Get clusters from K-means
    kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred1 = kmeans1.fit_predict(z1.data.cpu().numpy())

    kmeans2 = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred2 = kmeans2.fit_predict(z2.data.cpu().numpy())

    print("Initial Cluster Centers from view1: ", y_pred1)
    acc1, f1 = np.round(cluster_acc(y, y_pred1), 5)
    nmi1 = np.round(nmi_score(y, y_pred1), 5)
    ari1 = np.round(ari_score(y, y_pred1), 5)
    ami1 = np.round(ami_score(y, y_pred1), 5)
    print('Start-1: acc=%.5f, nmi=%.5f, ari=%.5f, ami=%.5f' % (acc1, nmi1, ari1, ami1))

    print("Initial Cluster Centers from view2: ", y_pred2)
    acc2, f1 = np.round(cluster_acc(y, y_pred2), 5)
    nmi2 = np.round(nmi_score(y, y_pred2), 5)
    ari2 = np.round(ari_score(y, y_pred2), 5)
    ami2 = np.round(ami_score(y, y_pred2), 5)
    print('Start-2: acc=%.5f, nmi=%.5f, ari=%.5f, ami=%.5f' % (acc2, nmi2, ari2, ami2))

    # Initialize D1 and D2
    D1 = Initialization_D(z1, y_pred1, args.n_clusters, args.d)
    D1 = torch.tensor(D1).to(torch.float32)

    D2 = Initialization_D(z2, y_pred2, args.n_clusters, args.d)  # 隐层为95 hidden为9631×95(num_sample*z)
    D2 = torch.tensor(D2).to(torch.float32)

    # s_tilde = np.full((args.num_sample, args.n_clusters), 1 / args.n_clusters, dtype=np.float32)  # s_tilde的初始化
    model.D1.data = D1.to(args.device)
    model.D2.data = D2.to(args.device)

    model.train()

    pbar = tqdm.tqdm(range(args.epoch), ncols=200)
    for epoch in pbar:
        x_hat1, x_hat2, s1, s2, p1, p2, z1, z2, z1_agcn, z2_agcn = model(x1, A1, x2, A2)
        if torch.any(p1==0):
            p1 = torch.clamp(p1, min=1e-20)
        if torch.any(p2==0):
            p2 = torch.clamp(p2, min=1e-20)

        L_REC = F.mse_loss(x_hat1, x1) + F.mse_loss(x_hat2, x2)

        # Constraints of D
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()

        loss_d1_D1 = d_cons1(model.D1, args.alpha)
        loss_d1_D2 = d_cons1(model.D2, args.alpha)
        loss_d1 = loss_d1_D1 + loss_d1_D2

        loss_d2_D1 = d_cons2(model.D1, args.d, args.n_clusters, args.alpha)
        loss_d2_D2 = d_cons2(model.D2, args.d, args.n_clusters, args.alpha)
        loss_d2 = loss_d2_D1 + loss_d2_D2

        L_D = loss_d1 + loss_d2

        if args.first_view == 'RNA':
            if epoch % 40 < 20:
                L_KL_s1 = distribution_loss(s1, refined_subspace_affinity(s1.data))
                L_KL_s2 = distribution_loss(s2, refined_subspace_affinity(s1.data))
                L_KL_p1 = distribution_loss(p1, refined_subspace_affinity(s1.data))
                L_KL_p2 = distribution_loss(p2, refined_subspace_affinity(s1.data))
            else:
                L_KL_s1 = distribution_loss(s1, refined_subspace_affinity(s2.data))
                L_KL_s2 = distribution_loss(s2, refined_subspace_affinity(s2.data))
                L_KL_p1 = distribution_loss(p1, refined_subspace_affinity(s2.data))
                L_KL_p2 = distribution_loss(p2, refined_subspace_affinity(s2.data))
        else:
            if epoch % 40 < 20:
                L_KL_s1 = distribution_loss(s1, refined_subspace_affinity(s2.data))
                L_KL_s2 = distribution_loss(s2, refined_subspace_affinity(s2.data))
                L_KL_p1 = distribution_loss(p1, refined_subspace_affinity(s2.data))
                L_KL_p2 = distribution_loss(p2, refined_subspace_affinity(s2.data))
            else:
                L_KL_s1 = distribution_loss(s1, refined_subspace_affinity(s1.data))
                L_KL_s2 = distribution_loss(s2, refined_subspace_affinity(s1.data))
                L_KL_p1 = distribution_loss(p1, refined_subspace_affinity(s1.data))
                L_KL_p2 = distribution_loss(p2, refined_subspace_affinity(s1.data))

        loss = L_REC + L_D + args.beta * ((L_KL_s1 + L_KL_s2) + (L_KL_p1 + L_KL_p2))

        # Evaluate clustering performance
        ari_s, nmi_s, ami_s, acc_s, f1_s, y_pred_s = assignment((s1 + s2).data, y)
        ari_p, nmi_p, ami_p, acc_p, f1_p, y_pred_p = assignment((p1 + p2).data, y)

        if nmi_p > args.nmi:
            args.acc = acc_p
            args.nmi = nmi_p
            args.ari = ari_p
            args.ami = ami_p
            args.f1 = f1_p

        pbar.set_postfix(
            {'loss': '{0:1.4f}'.format(loss), 'ari_s': '{0:1.4f}'.format(ari_s), 'nmi_s': '{0:1.4f}'.format(nmi_s),
             'ari_p': '{0:1.4f}'.format(ari_p), 'nmi_p': '{0:1.4f}'.format(nmi_p), 'f1_p': '{0:1.4f}'.format(f1_p)})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pbar.close()
    y_pred_last = y_pred_p

    acc, f1 = np.round(cluster_acc(y, y_pred_last), 5)
    nmi = np.round(nmi_score(y, y_pred_last), 5)
    ari = np.round(ari_score(y, y_pred_last), 5)
    ami = np.round(ami_score(y, y_pred_last), 5)

    print('Final: acc=%.5f, nmi=%.5f, ari=%.5f, ami=%.5f, f1=%.5f' % (acc, nmi, ari, ami, f1))

    end = time.time()
    print('Running time: ', end - start)

    z = torch.cat((z1_agcn, z2_agcn), dim=1)
    np.save('./output/{}/seed{}_label.npy'.format(args.dataset, args.seed), y_pred_p)
    np.save('./output/{}/seed{}_z.npy'.format(args.dataset, args.seed), z.cpu().detach().numpy())
    return acc, nmi, ari, ami


if __name__ == "__main__":
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    setup_seed(args.seed)
    dataset = LoadDataset(args.dataset, omic2='ADT') # 'omic2'(ADT/ATAC) is modified based on the second omics of the dataset.
    x1 = torch.tensor(dataset.x1, device=args.device)
    x2 = torch.tensor(dataset.x2, device=args.device)

    A1 = dataset.A1
    A2 = dataset.A2
    A1 = numpy_to_torch(A1, sparse=True).to(args.device)
    A2 = numpy_to_torch(A2, sparse=True).to(args.device)
    y = dataset.y

    args.num_sample = x1.shape[0]
    args.n_clusters = len(np.unique((y)))
    args.n_z = args.n_clusters * args.d

    ae1 = AE(ae_n_enc_1=args.ae_n_enc_1, ae_n_enc_2=args.ae_n_enc_2,
        ae_n_dec_1=args.ae_n_dec_1, ae_n_dec_2=args.ae_n_dec_2,
        n_input=args.n_d1, n_z=args.n_z).to(args.device)

    ae2 = AE(ae_n_enc_1=args.ae_n_enc_1, ae_n_enc_2=args.ae_n_enc_2,
        ae_n_dec_1=args.ae_n_dec_1, ae_n_dec_2=args.ae_n_dec_2,
        n_input=args.n_d2, n_z=args.n_z).to(args.device)

    agcn1 = DIF_MSIF_GCN(n_d=args.n_d1)
    agcn2 = DIF_MSIF_GCN(n_d=args.n_d2)

    model = scMISP(ae1, ae2, agcn1, agcn2)
    model = model.to(args.device)

    acc, nmi, ari, ami = train(model, x1, A1, x2, A2, y)