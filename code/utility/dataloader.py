import numpy as np
import os
import scipy.sparse as sp
import tqdm
import warnings

warnings.filterwarnings('ignore')


class Data(object):
    def __init__(self, path):
        self.path = path
        self.num_users = 0
        self.num_items = 0
        self.num_nodes = 0
        self.num_train = 0
        self.num_test = 0
        self.user_degree = 0
        self.item_degree = 0
        self.pos_length = None
        self.train_user = None
        self.test_user = None
        self.train_item = None
        self.test_item = None
        self.bipartite_graph = None
        self.user_item_net = None
        self.all_positive = None
        self.test_dict = None
        self.load_data()

    def load_data(self):
        # train_path = self.path + "/0.3noise_train.txt"
        train_path = self.path + "/train.txt"
        test_path = self.path + "/test.txt"

        print("1.Loading train and test data:")
        print("\t1.1 Loading train dataset:")
        train_user, self.train_user, self.train_item, self.num_train, self.pos_length = self.read_ratings(train_path)
        print("\t\tTrain dataset loading completed.")
        print("\t1.2 Loading test dataset:")
        test_user, self.test_user, self.test_item, self.num_test, _ = self.read_ratings(test_path)
        print("\t\tTest dataset loading completed.")
        print("\tTrain and test dataset loading completed.")
        self.num_users = max(train_user)
        self.num_items = max(self.train_item)

        self.num_users += 1
        self.num_items += 1

        self.num_nodes = self.num_users + self.num_items

        self.data_statistics()

        print("2.Construct user-item bipartite graph: (based on numpy.ndarray)")
        assert len(self.train_user) == len(self.train_item)
        self.user_item_net = sp.csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                           shape=(self.num_users, self.num_items))

        self.user_degree = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.user_degree[self.user_degree == 0.] = 1.
        self.item_degree = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.item_degree[self.item_degree == 0.] = 1.

        self.user_degree = np.power(self.user_degree, -0.5)
        self.item_degree = np.power(self.item_degree, -0.5)

        print("\t Bipartite graph constructed.")

        print("3.Construct adjacency matrix of graph:")
        # self.sparse_adjacency_matrix()
        # self.read_similarity_user_list(self.path + "/simi.txt")

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))
        self.test_dict = self.build_test()

    #   self.split_test_dict, self.split_state = self.create_sparsity_split()

    def read_ratings(self, file_name):
        inter_users, inter_items, unique_users = [], [], []
        inter_num = 0
        pos_length = []
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]
                unique_users.append(user_id)

                # if len(pos_id) < 1:
                #     print(user_id, pos_id)
                #     break
                # self.num_users = max(self.num_users, user_id)
                # self.num_items = max(self.num_items, max(pos_id))

                inter_users.extend([user_id] * len(pos_id))
                pos_length.append(len(pos_id))
                inter_items.extend(pos_id)
                inter_num += len(pos_id)
                line = f.readline()

        return np.array(unique_users), np.array(inter_users), np.array(inter_items), inter_num, pos_length

    def data_statistics(self):
        """ 输出读取数据的基本信息 """
        print("\tnum_users:", self.num_users)
        print("\tnum_items:", self.num_items)
        print("\tnum_nodes:", self.num_nodes)
        print("\tnum_train:", self.num_train)
        print("\tnum_test:", self.num_test)
        print("\tsparisty:", 1 - (self.num_train + self.num_test) / self.num_users / self.num_items)

    def sparse_adjacency_matrix(self):
        if self.bipartite_graph is None:
            try:
                pre_adjacency = sp.load_npz(self.path + '/pre_adj_mat.npz')
                print("\t Adjacency matrix loading completed.")
                norm_adjacency = pre_adjacency
            except:
                adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
                adjacency_matrix = adjacency_matrix.tolil()
                R = self.user_item_net.todok()

                degree_R = R / R.sum(axis=1)
                degree_R[np.isinf(degree_R)] = 0.
                # degree_R = sp.csr_matrix(degree_R)
                # sp.save_npz(self.path + '/pre_R_mat.npz', degree_R)
                '''
                    [ 0  R]
                    [R.T 0]
                '''
                # adjacency_matrix[:self.num_users, :self.num_users] = user_matrix
                adjacency_matrix[:self.num_users, self.num_users:] = R
                adjacency_matrix[self.num_users:, :self.num_users] = R.T
                adjacency_matrix = adjacency_matrix.tocsr()
                # csr_matrix = adjacency_matrix.tocsr()
                # sp.save_npz("./data/0.3yelpui.npz", adjacency_matrix)
                # adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

                row_sum = np.array(adjacency_matrix.sum(axis=1))
                d_inv = np.power(row_sum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                degree_matrix = sp.diags(d_inv)

                # D^(-1/2) A D^(-1/2)
                norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
                # sp.save_npz(self.path + '/pre_adj_mat.npz', norm_adjacency)
                print("\t Adjacency matrix constructed.")

            self.bipartite_graph = norm_adjacency

        return self.bipartite_graph

    def sparse_adjacency_matrix_norm(self):
        if self.bipartite_graph is None:
            try:
                pre_adjacency = sp.load_npz(self.path + '/pre_adj_mat_dual.npz')
                print("\t Adjacency matrix loading completed.")
                final_norm_adjacency = pre_adjacency
            except:
                adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
                adjacency_matrix = adjacency_matrix.tolil()
                R = self.user_item_net.tolil()

                degree_R = R / R.sum(axis=1)
                degree_R[np.isinf(degree_R)] = 0.

                degree_R = sp.csr_matrix(degree_R)
                sp.save_npz(self.path + '/pre_R_mat.npz', degree_R)

                '''
                    [ 0  R]
                    [R.T 0]
                '''
                adjacency_matrix[:self.num_users, self.num_users:] = R
                adjacency_matrix[self.num_users:, :self.num_users] = R.T

                adjacency_matrix = adjacency_matrix.todok()

                norm_adjacency = self.get_norm_adjacency(adjacency_matrix)
                print("\t Adjacency matrix constructed.")

                dual_adjacency_matrix = adjacency_matrix.dot(adjacency_matrix)
                print("\t A^2 constructed.")

                dual_adjacency_matrix = dual_adjacency_matrix.todok()

                dual_norm_adjacency = self.get_norm_adjacency(dual_adjacency_matrix)
                print("\t Dual_Adjacency matrix constructed.")

                final_norm_adjacency = norm_adjacency + dual_norm_adjacency

                sp.save_npz(self.path + '/pre_adj_mat_dual.npz', final_norm_adjacency)

                print("\t Final Adjacency matrix constructed.")

            self.bipartite_graph = final_norm_adjacency

        return self.bipartite_graph

    def get_norm_adjacency(self, adjacency_matrix):
        print(1)
        row_sum = np.array(adjacency_matrix.sum(axis=1))
        print(2)
        d_inv = np.power(row_sum, -0.5).flatten()
        print(3)
        d_inv[np.isinf(d_inv)] = 0.
        print(4)
        degree_matrix = sp.diags(d_inv)

        # D^(-1/2) A D^(-1/2)
        print(5)
        norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
        return norm_adjacency

    def sample_data_to_train_all(self):
        users = np.random.randint(0, self.num_users, len(self.train_user))
        sample_list = []
        for i, user in enumerate(users):
            positive_items = self.all_positive[user]
            if len(positive_items) == 0:
                continue
            positive_index = np.random.randint(0, len(positive_items))
            positive_item = positive_items[positive_index]
            while True:
                negative_item = np.random.randint(0, self.num_items)
                if negative_item in positive_items:
                    continue
                else:
                    break
            sample_list.append([user, positive_item, negative_item])

        return np.array(sample_list)

    def get_user_pos_items(self, users):
        positive_items = []
        for user in users:
            positive_items.append(self.user_item_net[user].nonzero()[1])
        return positive_items

    def get_user_simi_users(self, users):
        simi_users, simi_scores = [], []
        for user in users:
            simi_users.append(self.similarity_list[user][0])
            simi_scores.append(self.similarity_list[user][1])
        return simi_users, simi_scores

    def build_test(self):
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def read_similarity_user_list(self, file_name):
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                line = f.readline()
                while line is not None and line != "":
                    temp = line.strip()
                    arr = [int(i) for i in temp.split(" ")]
                    self.similarity_list[int(arr[0])] = arr[1:]
                    line = f.readline()
        else:
            R = self.user_item_net.tolil()
            simi_matrix = R * R.T
            self.get_similarity_user_list(file_name, simi_matrix.tocoo())

    def get_similarity_user_list(self, file_name, matrix):
        matrix.data[matrix.row == matrix.col] = 0

        for user in tqdm(range(self.num_users)):
            self.similarity_list[user] = []
            sub_matrix_col = matrix.col[matrix.row == user]
            sub_matrix_data = matrix.data[matrix.row == user]

            simi_list = np.argsort(sub_matrix_data)
            for i in range(1, 2):
                simi_id = sub_matrix_col[simi_list[-i]]
                if simi_id == user:
                    continue
                self.similarity_list[user].append(str(simi_id))
        print(self.similarity_list)
        with open(file_name, "w") as f:
            for i in self.similarity_list:
                strs = str(i) + " " + ' '.join(self.similarity_list[i])
                f.write(strs + "\n")
        f.close()

    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = dict()

        for uid in all_users:
            train_iids = self.all_positive[uid]
            test_iids = self.test_dict[uid]

            num_iids = len(train_iids) + len(test_iids)

            if num_iids not in user_n_iid.keys():
                user_n_iid[num_iids] = [uid]
            else:
                user_n_iid[num_iids].append(uid)

        split_uids = list()
        temp = []
        count = 1
        fold = 4
        n_count = self.num_train + self.num_test
        n_rates = 0
        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.num_train + self.num_test):
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state