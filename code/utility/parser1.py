import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN")

    parser.add_argument("--seed", type=int, default=2023, help="random seed for init")
    parser.add_argument("--model", default="LightGCN", help="Model Name")
    parser.add_argument(
        "--dataset",
        default="Movielens",
        help="Dataset to use, default: Movielens",
    )
    parser.add_argument("--mu_u", type=float, default=0.1, help="mu_u")
    parser.add_argument("--mu_i", type=float, default=0.1, help="mu_i")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning Rate")
    parser.add_argument('--dim', type=int, default=128, help='embedding size')
    parser.add_argument('--topK', nargs='?', default='[20]', help='size of Top-K')
    parser.add_argument("--in_size", default=128, type=int, help="Initial dimension size for entities.")
    parser.add_argument("--out_size", default=128, type=int, help="Output dimension size for entities.")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=300, help='batch size')
    parser.add_argument('--GCNLayer', type=int, default=2, help="the layer number of GCN")
    parser.add_argument("--data_path", nargs="?", default="../data/", help="Input data path.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")
    parser.add_argument("--verbose", type=int, default=5, help="Test interval")
    parser.add_argument("--num_gat_layer", type=int, default=1, help="Test interval")
    parser.add_argument('--regs', nargs='?', default='[1e-4]', help='Regularizations.')
    parser.add_argument('--k1', type=float, default=1e-4, help='user-user view weight')
    parser.add_argument('--k2', type=float, default=1e-4, help='item-item view weight')
    parser.add_argument("--device", type=str, default="cuda:0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    parser.add_argument("--num_heads", default=1, type=int, help="Number of attention heads")

    parser.add_argument("--multicore", type=int, default=0, help="use multiprocessing or not in test")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout", type=bool, default=False, help="consider node dropout or not")
    parser.add_argument("--mess_keep_prob", nargs='?', default='[0.1, 0.1, 0.1]', help="ratio of node dropout")
    parser.add_argument("--node_keep_prob", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='l2 regularization weight')
    parser.add_argument("--num_workers",type=int,default=10,help="Number of processes to construct batches")
    parser.add_argument('--l2', type=float, default=1e-2, help='l2 regularization weight')


    return parser.parse_args()