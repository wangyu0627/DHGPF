import torch
import time
import utility.parser1
import utility.batch_test
from utility.dataloader import Data
from model.train_model import GSLRec
import scipy.sparse as sp

def main():
    args = utility.parser1.parse_args()
    utility.batch_test.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load train and test
    dataset = Data(args.data_path + args.dataset)
    # load model
    if args.dataset == 'Movielens':
        uu_graph = sp.load_npz('../data/movielens_user.npz')
        ii_graph = sp.load_npz('../data/movielens_item.npz')
    elif args.dataset == 'Amazon':
        uu_graph = sp.load_npz('../data/amazon_user.npz')
        ii_graph = sp.load_npz('../data/amazon_item.npz')
    elif args.dataset == 'Douban Book':
        uu_graph = sp.load_npz('../data/dbook_user.npz')
        ii_graph = sp.load_npz('../data/dbook_item.npz')
    model = GSLRec(args, dataset, uu_graph, ii_graph, mu_u=args.mu_u, mu_i=args.mu_i, device=device)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_report_recall = 0.
    best_report_ndcg = 0.
    best_report_epoch = 0
    early_stop = 0
    for epoch in range(args.epochs):
        since = time.time()
        # Training and validation using a full graph
        if epoch % args.verbose == 0 or epoch > 50:
            result = utility.batch_test.Test(dataset, model, device, eval(args.topK), args.multicore,
                                             args.test_batch_size, long_tail=False)
            if result['recall'][0] > best_report_recall:
                early_stop = 0
                best_report_epoch = epoch + 1
                best_report_recall = result['recall'][0]
                best_report_ndcg = result['ndcg'][0]
            else:
                early_stop += 1

            if early_stop >= 10:
                print("early stop! best epoch:", best_report_epoch, "bset_recall:", best_report_recall, ',best_ndcg:',
                      best_report_ndcg)
                with open('./result/' + args.dataset + "/result.txt", "a") as f:
                    f.write(str(args.model) + " ")
                    f.write(str(args.lr) + " ")
                    f.write(str(best_report_epoch) + " ")
                    f.write(str(best_report_recall) + " ")
                    f.write(str(best_report_ndcg) + "\n")
                break
            else:
                print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])

        model.train()
        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        users, pos_items, neg_items = utility.batch_test.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // args.batch_size + 1
        average_loss = 0.
        average_reg_loss = 0.

        for batch_i, (batch_users, batch_positive, batch_negative) in enumerate(
                utility.batch_test.mini_batch(users, pos_items, neg_items, batch_size=args.batch_size)):
            batch_mf_loss, batch_emb_loss = model.bpr_loss(batch_users, batch_positive, batch_negative)
            batch_emb_loss = eval(args.regs)[0] * batch_emb_loss
            batch_loss = batch_emb_loss + batch_mf_loss
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            average_loss += batch_mf_loss.item()
            average_reg_loss += batch_emb_loss.item()

        average_loss = average_loss / num_batch
        average_reg_loss = average_reg_loss / num_batch
        time_elapsed = time.time() - since
        print("\t Epoch: %4d| train time: %.3f | train_loss:%.4f + %.4f" % (
            epoch + 1, time_elapsed, average_loss, average_reg_loss))

    print("best epoch:", best_report_epoch)
    print("best recall:", best_report_recall)
    print("best ndcg:", best_report_ndcg)
    # 创建字符串
    formatted_string = "mu_u:{}//mu_i:{}//best recall:{:.4f}//best ndcg:{:.4f}".format(
        args.mu_u, args.mu_i, best_report_recall, best_report_ndcg)
    # 打开文件并写入字符串
    with open('results_movielens.txt', 'a') as file:
        file.write(formatted_string + '\n')

if __name__ == '__main__':
    main()