import numpy as np
import scipy.sparse as sp
import torch
import os

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_movielens():
    path = "../data/Movielens/mp_matrix/"
    umu = sp.load_npz(path + "umu.npz")
    uau = sp.load_npz(path + "uau.npz")
    uou = sp.load_npz(path + "uou.npz")
    umgmu = sp.load_npz(path + "umgmu.npz")
    # ummmu = sp.load_npz(path + "ummmu.npz")
    mum = sp.load_npz(path + "mum.npz")
    mgm = sp.load_npz(path + "mgm.npz")
    # muuum = sp.load_npz(path + "muuum.npz")

    umu = sparse_mx_to_torch_sparse_tensor(normalize_adj(umu))
    uau = sparse_mx_to_torch_sparse_tensor(normalize_adj(uau))
    uou = sparse_mx_to_torch_sparse_tensor(normalize_adj(uou))
    umgmu = sparse_mx_to_torch_sparse_tensor(normalize_adj(umgmu))
    # ummmu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ummmu))
    mum = sparse_mx_to_torch_sparse_tensor(normalize_adj(mum))
    mgm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mgm))
    # muuum = sparse_mx_to_torch_sparse_tensor(normalize_adj(muuum))

    user_key = "user"
    item_key = "movie"
    meta_paths = {
        "user": [["umu"], ["uau"], ["uou"], ["umgmu"]],
        # , ["uau"], ["ummmu"], ["umgmu"] ["uou"],
        "movie": [["mum"], ["mgm"]],
        # , ["muuum"]
    }
    #     ,uau,ummmu,umgmuuou,
    # ,muuum
    return meta_paths,user_key,item_key,[umu,uau,uou,umgmu],[mum,mgm]

def load_amazon():
    path = "../data/Amazon/mp_matrix/"
    uiu = sp.load_npz(path + "uiu.npz")
    uibiu = sp.load_npz(path + "uibiu.npz")
    uiviu = sp.load_npz(path + "uiviu.npz")
    iui = sp.load_npz(path + "iui.npz")
    ibi = sp.load_npz(path + "ibi.npz")
    ici = sp.load_npz(path + "ici.npz")
    ivi = sp.load_npz(path + "ivi.npz")


    uiu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uiu))
    uibiu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uibiu))
    uiviu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uiviu))
    iui = sparse_mx_to_torch_sparse_tensor(normalize_adj(iui))
    ibi = sparse_mx_to_torch_sparse_tensor(normalize_adj(ibi))
    ici = sparse_mx_to_torch_sparse_tensor(normalize_adj(ici))
    ivi = sparse_mx_to_torch_sparse_tensor(normalize_adj(ivi))

    user_key = "user"
    item_key = "item"
    meta_paths = {
        "user": [["uiu"], ["uibiu"], ["uiviu"]],
        # , ["uiviu"]
        "item": [["iui"], ["ici"], ["ibi"], ["ivi"]],
        # , ["ivi"], ["ici"]
    }
    #     ,uau,ummmu,umgmu
    # ,muuum
    return meta_paths,user_key,item_key,[uiu,uibiu,uiviu],[iui,ici,ibi,ivi]

def load_douban_book():
    path = "../data/Douban Book/mp_matrix/"
    ubu = sp.load_npz(path + "ubu.npz")
    ugu = sp.load_npz(path + "ugu.npz")
    ulu = sp.load_npz(path + "ulu.npz")
    # ubabu = sp.load_npz(path + "ubabu.npz")
    # ubpbu = sp.load_npz(path + "ubpbu.npz")
    bab = sp.load_npz(path + "bab.npz")
    bpb = sp.load_npz(path + "bpb.npz")
    byb = sp.load_npz(path + "byb.npz")
    # bugub = sp.load_npz(path + "bugub.npz")


    ubu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ubu))
    ugu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ugu))
    ulu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ulu))
    # ubabu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ubabu))
    # ubpbu = sparse_mx_to_torch_sparse_tensor(normalize_adj(ubpbu))
    bab = sparse_mx_to_torch_sparse_tensor(normalize_adj(bab))
    bpb = sparse_mx_to_torch_sparse_tensor(normalize_adj(bpb))
    byb = sparse_mx_to_torch_sparse_tensor(normalize_adj(byb))
    # bugub = sparse_mx_to_torch_sparse_tensor(normalize_adj(bugub))


    user_key = "user"
    item_key = "book"
    meta_paths = {
        "user": [["ubu"], ["ugu"], ["ulu"]],
        # , ["uiviu"], ["ubabu"], ["ubpbu"]
        "book": [["bab"], ["bpb"], ["byb"]],
        # , ["ivi"], ["ici"], ["bugub"]
    }
    #     ,uau,ummmu,umgmu
    # ,muuum,ubabu,ubpbu,bugub
    return meta_paths,user_key,item_key,[ubu,ugu,ulu],[bab,bpb,byb]

def load_data(dataset):
    if dataset == "Movielens":
        data = load_movielens()
    elif dataset == "Amazon":
        data = load_amazon()
    else:
        data = load_douban_book()
    return data

# if __name__ == '__main__':
#     load_movielens()