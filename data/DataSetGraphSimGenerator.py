import configparser
import json
import logging
import os
import pandas
import random
import sys
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import pickle

import torch
from gensim.models import KeyedVectors
from torch import nn
from torch.utils.data import Dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from kb_algorithm.event_w2v.event_w2v import Event2Vec
# from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DataSetGraphSimGenerator():
    def __init__(self, data_set_id=1, dataset_version=None):
        self.data_set_id = data_set_id
        self.dataset_name = "train_ticket" if data_set_id == 1 else "sock_shop"
        self.dataset_name = "train_ticket"
        logging.warning("graph_sim_dataset: {}".format(self.dataset_name))
        self.kb_dir_path = os.path.join(os.path.dirname(__file__), "kb_construction", "event_graph_data", self.dataset_name)
        self.online_dir_path = os.path.join(os.path.dirname(__file__), "kb_online_construct", "event_graph_data", self.dataset_name)

        # 最终 所有图相似算法所需数据得存储位置
        dataset_special = "{}_{}".format(self.dataset_name, dataset_version) if dataset_version else self.dataset_name
        # dataset_special = "sock_shop_final_same"
        dataset_special = "train_ticket_final_same"
        self.graph_sim_data_dir = os.path.join(os.path.dirname(__file__), "data", "data_for_graph_sim", dataset_special)
        if not os.path.exists(self.graph_sim_data_dir):
            os.makedirs(self.graph_sim_data_dir)

        self.e2v = None
        self.kb_graph_data = {}
        self.online_graph_data = {}
        self.label_data_path = os.path.join(self.graph_sim_data_dir, "labeled_data.json")
        self.label_graph_class_path = os.path.join(self.graph_sim_data_dir, "labeled_graph_class_data.json")
        self.pickle_data_saved_dir = os.path.join(self.graph_sim_data_dir, "pickle_data")
        if not os.path.exists(self.pickle_data_saved_dir):
            os.makedirs(self.pickle_data_saved_dir)
        pass

    def __generate_dir_pickle(self, file_path, state, error_type, time_ts):
        """
        生成每一个nodes_relations.json对应的pickle文件, 其中会加载w2v获取词向量
        :param file_path: nodes_relations。json路径
        :param state: 线上还是线下 即 kb 还是 online
        :param error_type: 错误类型
        :param time_ts: 时间戳
        :return:
        """
        data_info = read_json_data(file_path)

        node_ids = [_["id"] for _ in data_info["nodes_dict"]]
        node_ids.sort()
        assert len(set(node_ids)) == len(node_ids)
        node_id_index = {id: index for index, id in enumerate(node_ids)}
        node_index_id = node_ids
        node_id_value = dict()
        for _ in data_info["nodes_dict"]:
            if "value" in _.keys():
                node_id_value[_["id"]] = _["value"]
            else:
                node_id_value[_["id"]] = "without"
        node_index_value = [node_id_value[node_index_id[_]] for _ in range(len(node_ids))]
        assert len(node_index_value) == len(node_id_value) == len(node_index_id) == len(node_id_index)
        # 生成邻接矩阵
        adjacency_matrix = np.zeros((len(node_ids), len(node_ids)))
        for r in data_info["relations_dict"]:
            start_index = node_id_index[r["_start_node_id"]]
            end_index = node_id_index[r["_end_node_id"]]
            adjacency_matrix[start_index][end_index] = 1
            assert adjacency_matrix[end_index][start_index] == 0
        logging.info("generate_data:  nodes:{} = {}  relations:{} = {}".format(len(node_ids), data_info["node_num"],
                                                                               int(adjacency_matrix.sum()),
                                                                               data_info["relation_num"]))
        assert len(node_ids) == data_info["node_num"] == len(set(node_ids)) and int(adjacency_matrix.sum()) == data_info["relation_num"]
        # 转为稀疏矩阵
        loc = np.where(adjacency_matrix == 1)
        adj_sparse = sp.csr_matrix((np.ones(loc[0].shape), (loc[0], loc[1])), shape=adjacency_matrix.shape,
                                   dtype=np.int8)

        # 对应的初始特征向量，即word2vec
        if self.e2v is None:
            self.e2v = Event2Vec(data_set_id=self.data_set_id)
            self.e2v.load_model()
        # words = [self.e2v.event_word[_] for _ in node_ids]
        words = list()
        for _ in node_ids:
            if _ not in self.e2v.event_word.keys():
                logging.error("no match word! event:{}".format(_))
                words.append("")
            else:
                word = self.e2v.event_word[_]
                words.append(word)
        # word_arrays = [self.e2v.model[word] for word in words]
        word_arrays = list()
        for word in words:
            # logging.info("get vec of word :{}".format(word))
            if word not in self.e2v.model:
                logging.error("word no vec! word:{} event:{}".format(word,
                                                                       self.e2v.word_event[word]
                                                                       if word in self.e2v.word_event.keys() else ""))
                word_arrays.append(np.zeros(self.e2v.model.vector_size))
            else:
                word_array = self.e2v.model[word]
                word_arrays.append(word_array)

        features = np.stack(word_arrays)
        logging.info("file_path:{}".format(file_path))
        logging.info("features:{} adj:{}".format(features.shape, adj_sparse.todense().shape))
        data = dict(
            data_set_type=data_info["data_set_type"],
            fault_type=data_info["fault_type"],
            node_num=data_info["node_num"],
            relation_num=data_info["relation_num"],
            nodes_dict=data_info["nodes_dict"],
            relations_dict=data_info["relations_dict"],

            node_id_index=node_id_index,
            node_index_id=node_index_id,
            node_id_value=node_id_value,
            node_index_value=node_index_value,

            adj_sparse=adj_sparse,
            adj=adj_sparse.toarray(),
            fetures=features,
        )

        # save_pickle_data(os.path.join(os.path.dirname(file_path), "adj_sparse.pickle"), data)
        file_name = "_".join([state, error_type, time_ts, "_adj_feature.pickle"])
        save_pickle_data(os.path.join(self.pickle_data_saved_dir, file_name), data)

        return os.path.join(self.pickle_data_saved_dir, file_name)


    def generate_dataset_example(self):
        # w2v词向量直接加载# http://www.linzehui.me/2018/08/19/%E7%A2%8E%E7%89%87%E7%9F%A5%E8%AF%86/%E5%85%B3%E4%BA%8EPytorch%E4%B8%ADEmbedding%E7%9A%84padding/
        tmp_file = ""
        wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
        vocab = wvmodel.vocab
        vocab_size = len(vocab) + 1
        embed_size = 300  # 维度需要和预训练词向量维度统一
        weight = torch.zeros(vocab_size + 1, embed_size)

        for i in range(len(wvmodel.index2word)):
            try:
                index = wvmodel.word_to_idx[wvmodel.index2word[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(wvmodel.get_vector(
                wvmodel.idx_to_word[wvmodel.word_to_idx[wvmodel.index2word[i]]]))

        # embed
        embedding = nn.Embedding.from_pretrained(weight)

    def generate_dataset_pickle(self):
        """获取所有 data_set_id所涉及的知识库文件  和 线上数据的pickle形式"""
        logging.error("generate_data_in:{}".format(self.graph_sim_data_dir))
        # 生成 知识库 和线下数据 对应的pickle文件，并保存 error_type:pickle_file_path 的字典
        for root, dirs, files in os.walk(self.kb_dir_path):
            for dir in dirs:
                error_type = dir
                file_path = os.path.join(self.kb_dir_path, dir, "nodes_relations.json")
                pickle_data_path = self.__generate_dir_pickle(file_path, "kb", error_type, "")
                self.kb_graph_data[error_type] = pickle_data_path.replace(os.path.dirname(__file__), "")

        for root, dirs, files in os.walk(self.online_dir_path):
            for dir in dirs:
                error_type = dir
                # if error_type in ["order_count_500", "net_delay_user_db_user_register_and_log_in_delay", "user_user_register_500", "order_cart_500",
                #                   "net_loss_cart_db_cart_add_cart_delay","k_order_db_order_order_disappeared", "order_payment_500"]:
                #     continue
                for r,d,f in os.walk(os.path.join(self.online_dir_path, dir)):
                    for pickle_name in d:
                        file_path = os.path.join(self.online_dir_path, dir, pickle_name, "nodes_relations.json")
                        pickle_data_path = self.__generate_dir_pickle(file_path, "online", error_type, pickle_name)
                        if error_type in self.online_graph_data.keys():
                            self.online_graph_data[error_type].add(pickle_data_path.replace(os.path.dirname(__file__), ""))
                        else:
                            self.online_graph_data[error_type] = set()
                            self.online_graph_data[error_type].add(pickle_data_path.replace(os.path.dirname(__file__), ""))

        # 线上和知识库数据 两两组合，并标注是否相似
        labeled_data = []
        total_num, pos_num, neg_num = 0, 0, 0
        for k_o, v_o in self.online_graph_data.items():
            v_o_list = list(v_o)
            total_num += len(v_o_list) * len(self.kb_graph_data)
            for k_f, v_f in self.kb_graph_data.items():
                if k_o == k_f:
                    # 相似
                    labeled_data.extend([(str(_), str(v_f), 1) for _ in v_o_list])
                    pos_num += len(v_o_list)
                    pass
                else:
                    # 不相似
                    labeled_data.extend([(str(_), str(v_f), 0) for _ in v_o_list])
                    neg_num += len(v_o_list)
        assert total_num == len(labeled_data)
        labels = np.array([_[2] for _ in labeled_data])
        assert pos_num == labels.sum() and neg_num == len(labels) - labels.sum()
        save_json_data(self.label_data_path, labeled_data)
        logging.warning("dataset:{} labeled_data_saved! pos:{} neg:{} loc:{}".format(self.dataset_name, pos_num, neg_num,
                                                                                     self.label_data_path))

        # 生成标注数据用于检测 图 分类是否正确
        error_name_list = list(self.kb_graph_data.keys())
        error_name_list.sort()
        online_info = []
        for error_name, time_set in self.online_graph_data.items():
            for pickle_name in list(time_set):
                online_info.append( [pickle_name, error_name, error_name_list.index(error_name)] )
        graph_class_data = dict(
            online_info_num=len(online_info),
            kb_error_num=len(error_name_list),
            online_info=online_info,
            kb_data=self.kb_graph_data,
            error_name_list=error_name_list
        )
        save_json_data(self.label_graph_class_path, graph_class_data)
        logging.warning("dataset:{} labeled_data_saved! online_info:{} error_name_list:{} loc:{}".format(
            self.dataset_name, len(online_info), len(error_name_list), self.label_graph_class_path))



    def generate_dataset_pickle_test(self):
        """获取模型测试数据  graph1_x_graph1"""
        # 生成 知识库 和线下数据 对应的pickle文件，并保存 error_type:pickle_file_path 的字典
        for root, dirs, files in os.walk(self.kb_dir_path):
            for dir in dirs:
                error_type = dir
                file_path = os.path.join(self.kb_dir_path, dir, "nodes_relations.json")
                pickle_data_path = self.__generate_dir_pickle(file_path, "kb", error_type, "")
                self.kb_graph_data[error_type] = pickle_data_path.replace(os.path.dirname(__file__), "")

        for root, dirs, files in os.walk(self.online_dir_path):
            for dir in dirs:
                error_type = dir
                for r,d,f in os.walk(os.path.join(self.online_dir_path, dir)):
                    for _ in d:
                        file_path = os.path.join(self.online_dir_path, dir, _, "nodes_relations.json")
                        pickle_data_path = self.__generate_dir_pickle(file_path, "online", error_type, _)
                        if error_type in self.online_graph_data.keys():
                            self.online_graph_data[error_type].add(pickle_data_path.replace(os.path.dirname(__file__), ""))
                        else:
                            self.online_graph_data[error_type] = set()
                            self.online_graph_data[error_type].add(pickle_data_path.replace(os.path.dirname(__file__), ""))

        # 线上和知识库数据 两两组合，并标注是否相似
        labeled_data = []
        total_num, pos_num, neg_num = 0, 0, 0
        for k_o, v_o in self.online_graph_data.items():
            v_o_list = list(v_o)
            for _v_o in v_o_list:
                labeled_data.append((str(_v_o), str(_v_o), 0))
                neg_num += 1
            for k_f, v_f in self.kb_graph_data.items():
                if k_o != k_f:
                    labeled_data.extend([(str(_), str(v_f), 1) for _ in v_o_list])
                    pos_num += len(v_o_list)
        labels = np.array([_[2] for _ in labeled_data])
        assert pos_num == labels.sum() and neg_num == len(labels) - labels.sum()
        save_json_data(self.label_data_path, labeled_data)
        logging.warning("dataset:{} labeled_data_saved! pos:{} neg:{} loc:{}".format(self.dataset_name, pos_num, neg_num,
                                                                                     self.label_data_path))

    def get_graph_pairs(self):
        label_data = read_json_data(self.label_data_path)
        for data in label_data:
            graph_online = read_pickle_data(os.path.dirname(__file__) + data[0])
            graph_kb = read_pickle_data(os.path.dirname(__file__) + data[1])
            assert graph_online["fetures"].shape[0] == graph_online["adj_sparse"].shape[0] == graph_online["adj_sparse"].shape[1]
            assert graph_kb["fetures"].shape[0] == graph_kb["adj_sparse"].shape[0] == graph_kb["adj_sparse"].shape[1]

            graph_1_info = (graph_online["fetures"], graph_online["adj_sparse"], graph_online["node_index_value"])
            graph_2_info = (graph_kb["fetures"], graph_kb["adj_sparse"], graph_kb["node_index_value"])
            label = [data[2]]

            yield (graph_1_info, graph_2_info, label)
        #
        # file_path = os.path.join(self.kb_dir_path, "f1", "adj_sparse.pickle")
        # data_info = read_pickle_data(file_path)
        # logging.warning("pickle_data! dataset:{} fault_type:{}".format(data_info["data_set_type"], data_info["fault_type"]))
        #
        # return torch.as_tensor([data_info["fetures"], data_info["adj_sparse"]]), (data_info["fetures"], data_info["adj_sparse"])


class CustomDataset(Dataset):#需要继承data.Dataset
    #https://blog.csdn.net/liuweiyuxiang/article/details/84037973
    def __init__(self, data_set_id, dataset_version=None, mode="train", max_node_num=100, repeat_pos_data=0, resplit=False,
                 random_expand_train_data_flag=False, dataset_dir=None):
        self.data_set_id = data_set_id
        self.dataset_name = "train_ticket" if data_set_id == 1 else "sock_shop"
        dataset_special = "{}_{}".format(self.dataset_name, dataset_version) if dataset_version else self.dataset_name
        self.dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "data_for_graph_sim",
                                        dataset_special)
        if dataset_dir:
            self.dataset_dir = dataset_dir
        self.train_data_path = os.path.join(self.dataset_dir, "train_labeled_data.json")
        self.test_data_path = os.path.join(self.dataset_dir, "test_labeled_data.json")
        self.validation_data_path = os.path.join(self.dataset_dir, "validation_labeled_data.json")
        self.label_data_path = os.path.join(self.dataset_dir, "labeled_data.json")
        self.label_graph_class_path = os.path.join(self.dataset_dir, "labeled_graph_class_data.json")

        self.max_node_num = max_node_num
        self.repeat_pos_data = repeat_pos_data
        self.mode = mode

        if resplit or not os.path.exists(self.train_data_path):
            self.split_labeled_data()
            if random_expand_train_data_flag:
                self.random_expand_train_data(expand_num=9, delete_part=0.1)
            if self.repeat_pos_data:
                self.guocaiyang_train()
        if mode == "train":
            self.labeled_data = read_json_data(self.train_data_path)
        elif mode == "test":
            self.labeled_data = read_json_data(self.test_data_path)
        elif mode == "val":
            self.labeled_data = read_json_data(self.validation_data_path)
        elif mode == "test_val":
            self.labeled_data = read_json_data(self.test_data_path) + read_json_data(self.validation_data_path)
        else:
            self.labeled_data = read_json_data(self.label_data_path)

    def __getitem__(self, index):
        online_path, kb_path, label = self.labeled_data[index]
        online_path = (os.path.dirname(__file__) + online_path).replace("\\", os.sep).replace("/", os.sep)
        kb_path = (os.path.dirname(__file__) + kb_path).replace("\\", os.sep).replace("/", os.sep)

        graph_online = read_pickle_data(online_path)
        graph_kb = read_pickle_data(kb_path)
        graph_online_feature, graph_online_A_list = self.process_graph(
            (graph_online["fetures"], graph_online["adj"], graph_online["node_index_value"]))
        graph_kb_feature, graph_kb_A_list = self.process_graph(
            (graph_kb["fetures"], graph_kb["adj"], graph_kb["node_index_value"]))

        sample = {
            'graph_online_adj': torch.as_tensor(graph_online_A_list, dtype=torch.float32, device=device),
            'graph_online_feature': torch.as_tensor(graph_online_feature, dtype=torch.float32, device=device),
            'graph_kb_adj': torch.as_tensor(graph_kb_A_list, dtype=torch.float32, device=device),
            'graph_kb_feature': torch.as_tensor(graph_kb_feature, dtype=torch.float32, device=device),
            'label': torch.as_tensor(label, dtype=torch.long, device=device)
        }
        return sample

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.labeled_data)

    def graph_class_data(self, use_best_path=False, graph_accuary=None, online_offline_proportion=None):
        """
        返回 一个online 图与所有 kb图 两两成对的数据
        :param use_best_path:
        :param graph_accuary: 小数 表示正确点数占比
        :param online_offline_proportion: 小数或整数 表示online:offline 's proportion
        :return:
        """
        online_path_in_mode = list(set([o_p for o_p, kb_p, label in self.labeled_data]))
        label_graph_class_data = read_json_data(self.label_graph_class_path)
        online_info = label_graph_class_data["online_info"]
        kb_data = label_graph_class_data["kb_data"]
        error_name_list = label_graph_class_data["error_name_list"]
        error_name_list.sort()
        for online_data_path, e_name, e_index in online_info:
            # 只加载该mode状态里的数据
            if online_data_path not in online_path_in_mode:
                continue
            online_path = (os.path.dirname(__file__) + online_data_path).replace("\\", os.sep).replace("/", os.sep)
            if use_best_path:
                online_path = online_path.replace(self.dataset_name, "{}_best".format(self.dataset_name))
            graph_online = read_pickle_data(online_path)

            # 根据图质量要求 控制online图
            if graph_accuary:
                logging.info("control init: graph_online({})".format(graph_online["fetures"].shape))
                compare = graph_accuary
                # 读取正确的点数
                labeled_info = read_json_data(os.path.join(os.path.dirname(__file__), "..", "data", "data_for_svm_classifier_online", "{}_labeled.json".format(e_name)))
                ids_right = labeled_info["ids"]
                save_node_num = int(len(ids_right) / graph_accuary)
                # 获取正确结点的索引 以及其他要保留结点的索引 生成 最终要保留的所有点的index
                index_saved = list()
                for id in ids_right:
                    if id not in graph_online["node_id_index"].keys():
                        logging.error("error:{} id:{} not in {}".format(e_name, id, online_data_path))
                        continue
                    index_saved.append(graph_online["node_id_index"][id])
                node_value_array = np.array(graph_online["node_index_value"])
                node_value_array[index_saved] = float("inf")
                # if save_other_node_num > 0:
                #     node_value_array = np.array(graph_online["node_index_value"])
                #     node_value_array[index_saved] = -1
                #     if save_node_num < len(node_value_array):
                #         index_other = np.argpartition(node_value_array, -save_other_node_num)[-save_other_node_num:]
                #         index_saved = np.concatenate((index_saved, index_other))
                #     else:
                #         index_saved = range(len(node_value_array))

                graph_online["fetures"], graph_online["adj"], graph_online["node_index_value"] = self.control_node_num(
                    (graph_online["fetures"], graph_online["adj"], node_value_array), diy_node_num=save_node_num
                )


            graph_online_feature, graph_online_A_list = self.process_graph(
                (graph_online["fetures"], graph_online["adj"], graph_online["node_index_value"]))
            graph_online_adj_list, graph_online_feature_list, graph_kb_adj_list = list(), list(), list()
            graph_kb_feature_list, label_list = list(), list()
            for e_kb_name in error_name_list:
                kb_data_path = kb_data[e_kb_name]
                label = 1.0 if e_name == e_kb_name else 0.
                kb_path = (os.path.dirname(__file__) + kb_data_path).replace("\\", os.sep).replace("/", os.sep)
                if use_best_path:
                    kb_path = kb_path.replace(self.dataset_name, "{}_best".format(self.dataset_name))
                graph_kb = read_pickle_data(kb_path)

                # online:offline 和 强行控制kb数目 [10, 8, 5, 2, 1, 0.8, 0.5, 0.3, 0.2, 0.1]
                if online_offline_proportion:
                    logging.info("control init: graph_kb({})".format(graph_kb["fetures"].shape))
                    compare = online_offline_proportion
                    online_node_num = graph_online["fetures"].shape[0]
                    offline_node_num = int(online_node_num / compare)
                    offline_node_num = 1 if offline_node_num <= 0 else offline_node_num
                    graph_kb["fetures"], graph_kb["adj"], graph_kb["node_index_value"] = self.control_node_num(
                        (graph_kb["fetures"], graph_kb["adj"], graph_kb["node_index_value"]), diy_node_num=offline_node_num
                    )
                    logging.info("control done: graph_kb({})".format(graph_kb["fetures"].shape))

                graph_kb_feature, graph_kb_A_list = self.process_graph(
                    (graph_kb["fetures"], graph_kb["adj"], graph_kb["node_index_value"]))
                graph_online_adj_list.append(torch.tensor(graph_online_A_list, dtype=torch.float32, device=device))
                graph_online_feature_list.append(torch.tensor(graph_online_feature, dtype=torch.float32, device=device))
                graph_kb_adj_list.append(torch.tensor(graph_kb_A_list, dtype=torch.float32, device=device))
                graph_kb_feature_list.append(torch.tensor(graph_kb_feature, dtype=torch.float32, device=device))
                label_list.append(torch.tensor(label, dtype=torch.long, device=device))
            sample = {
                'graph_online_adj': torch.stack(graph_online_adj_list),
                'graph_online_feature': torch.stack(graph_online_feature_list),
                'graph_kb_adj': torch.stack(graph_kb_adj_list),
                'graph_kb_feature': torch.stack(graph_kb_feature_list),
                'label': torch.stack(label_list),
                'error_name_list': error_name_list,
                'online_data_path': online_data_path,
            }
            assert torch.sum(sample["label"]).item() == 1.0

            yield sample, (online_data_path, e_name, e_index, error_name_list)

    def pos_neg_num(self):
        pos, neg = 0, 0
        for data in self.labeled_data:
            if data[2] == 1:
                pos += 1
            elif data[2] == 0:
                neg += 1
        return pos, neg

    def print_data_set_info(self):
        pos, neg = self.pos_neg_num()
        logging.error("dataset_id:{} mode:{} pos:{} neg:{}".format(self.data_set_id, self.mode,
                                                                   pos, neg))

    def random_expand_train_data(self, expand_num, delete_part=0.1):

        # 获取训练数据 中的online pickle ptha
        train_labeled_data = read_json_data(self.train_data_path)
        train_online_timepieces = list(set([data[0] for data in train_labeled_data]))
        df_raw = pandas.DataFrame(train_labeled_data, columns=["o_p", "f_p", "label"])
        # 对于每个pickle 随机删除一些点和 边
        for piece in train_online_timepieces:
            # 匹配的 kb 和不匹配的kb
            match_kb = list(df_raw[(df_raw["o_p"]==piece) & (df_raw["label"]==1)]["f_p"])[0]
            no_match_kb = list(df_raw[(df_raw["o_p"]==piece) & (df_raw["label"]==0)]["f_p"])

            online_path = (os.path.dirname(__file__) + piece).replace("\\", os.sep).replace("/", os.sep)
            graph_online = read_pickle_data(online_path)
            for index in range(expand_num):
                graph_online_copy = deepcopy(graph_online)
                new_pickle_path = online_path.replace(".pickle", "diy{}.pickle".format(index))

                index_delete = random.sample(range(graph_online_copy["node_num"]), int(graph_online_copy["node_num"] * (1-delete_part)))
                id_delete = [graph_online_copy["node_index_id"][index] for index in index_delete]

                # 删除 结点关系字典
                for node in graph_online["nodes_dict"]:
                    if node["id"] in id_delete:
                        graph_online_copy["nodes_dict"].remove(node)

                for relation in graph_online["relations_dict"]:
                    if (relation["_start_node_id"] in id_delete) or (relation["_end_node_id"] in id_delete):
                        graph_online_copy["relations_dict"].remove(relation)
                graph_online_copy["relation_num"] = len(graph_online_copy["relations_dict"])
                graph_online_copy["node_num"] = len(graph_online_copy["nodes_dict"])

                for id in id_delete:
                    del graph_online_copy["node_id_index"][id]
                    del graph_online_copy["node_id_value"][id]

                graph_online_copy["node_index_id"] = np.delete(graph_online_copy["node_index_id"], np.array(index_delete), axis=0).tolist()
                graph_online_copy["node_index_value"] = np.delete(graph_online_copy["node_index_value"], np.array(index_delete), axis=0).tolist()

                graph_online_copy["fetures"] = np.delete(graph_online_copy["fetures"], np.array(index_delete), axis=0)

                graph_online_copy["adj"] = np.delete(graph_online_copy["adj"], np.array(index_delete), axis=0)
                graph_online_copy["adj"] = np.delete(graph_online_copy["adj"], np.array(index_delete), axis=1)

                loc = np.where(graph_online_copy["adj"]==1)
                graph_online_copy["adj_sparse"] = sp.csr_matrix((np.ones(loc[0].shape), (loc[0], loc[1])), shape=graph_online_copy["adj"].shape,
                                           dtype=np.int8)

                save_pickle_data(new_pickle_path, graph_online_copy)
                replace_part = str(os.path.dirname(__file__))
                replace_part = replace_part.replace("\\", os.sep).replace("/", os.sep)
                save_pickle_path = new_pickle_path.replace(replace_part, "")
                train_labeled_data.append([save_pickle_path, match_kb, 1])
                for kb in no_match_kb:
                    train_labeled_data.append([save_pickle_path, kb, 0])

        df_expand = pandas.DataFrame(train_labeled_data, columns=["o_p", "f_p", "label"])
        assert (len(df_raw) * (expand_num+1)) == len(df_expand)
        save_json_data(self.train_data_path, train_labeled_data)
        if self.mode == "train":
            self.labeled_data = read_json_data(self.train_data_path)
        logging.warning("expand train data done! raw:{}+{}={} new{}+{}={} expand_num:{}".format(
            len(df_raw[df_raw["label"]==0]),
            len(df_raw[df_raw["label"]==1]),
            len(df_raw),
            len(df_expand[df_expand["label"]==0]),
            len(df_expand[df_expand["label"]==1]),
            len(df_expand),
            expand_num
        ))

    def guocaiyang_train(self):
        """过采样训练集正样本"""
        train_data = read_json_data(self.train_data_path)
        df_train = pandas.DataFrame(train_data, columns=["o_p", "kb_p", "label"])
        pos_train = df_train[df_train["label"]==1]
        neg_train = df_train[df_train["label"]==0]

        pos_train = self.repeat_df(pos_train, self.repeat_pos_data, len(neg_train) // len(pos_train))
        train = np.array(pandas.concat([pos_train, neg_train]).sample(frac=1)).tolist()
        save_json_data(self.train_data_path, train)

    def repeat_df(self, df_data, repeat_pos_data_flag, neg_pos):
        if repeat_pos_data_flag >= 1:
            if repeat_pos_data_flag == 1:
                # 自适应
                if neg_pos > 1:
                    df_data = pandas.concat([df_data] * neg_pos)
            else:
                df_data = pandas.concat([df_data] * neg_pos)
        return df_data

    def split_labeled_data(self):
        all_data = read_json_data(self.label_data_path)
        import pandas as pd

        def repeat_df(df_data, repeat_pos_data_flag, neg_pos):
            if repeat_pos_data_flag >= 1:
                if repeat_pos_data_flag == 1:
                    # 自适应
                    if neg_pos > 1:
                        df_data = pd.concat([df_data] * neg_pos)
                else:
                    df_data = pd.concat([df_data] * neg_pos)
            return df_data

        df = pd.DataFrame(all_data, columns=["o_p", "kb_p", "label"])

        # # 方式1 将online时间段随机分为6;2:2
        # online_pickle_names = list(set(df["o_p"]))
        # random.shuffle(online_pickle_names)
        # occupy = [0.6, 0.2, 0.2]
        # train_num, test_num, val_num = int(occupy[0] * len(online_pickle_names)), int(occupy[1] * len(online_pickle_names)), int(
        #     occupy[2] * len(online_pickle_names))
        # train_online_names = online_pickle_names[:train_num]
        # test_online_names = online_pickle_names[train_num:train_num + test_num]
        # val_online_names = online_pickle_names[train_num + test_num:]
        # 方式2 需要确保每种故障类型 都能 包含训练\测试\验证数据
        train_online_names, test_online_names, val_online_names = list(), list(), list()
        kb_unique_names = list(set(df["kb_p"]))
        for kb_name in kb_unique_names:
            new_df = df[(df["kb_p"]==kb_name) & (df["label"]==1)]
            kb_online_names = list(set(new_df["o_p"]))
            random.shuffle(kb_online_names)
            occupy = [0.6, 0.3, 0.1]
            train_num, test_num, val_num = int(occupy[0] * len(kb_online_names)), int(
                occupy[1] * len(kb_online_names)), int(
                occupy[2] * len(kb_online_names))
            train_l = kb_online_names[:train_num]
            test_l = kb_online_names[train_num:train_num + test_num]
            val_l = kb_online_names[train_num + test_num:]
            train_online_names.extend(train_l)
            test_online_names.extend(test_l)
            val_online_names.extend(val_l)
            logging.info(
                "kb_name:{}\n online_times:{} train:{} test:{} val:{}".format(kb_name, len(new_df), len(train_l),
                                                                              len(test_l), len(val_l)))

        df_train = pd.concat([df.loc[df["o_p"] == name] for name in train_online_names])
        df_test = pd.concat([df.loc[df["o_p"] == name] for name in test_online_names])
        df_val = pd.concat([df.loc[df["o_p"] == name] for name in val_online_names])
        # pos_df = repeat_df(pos_df, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]

        pos_train = df_train[df_train["label"] == 1]
        # pos_train = self.repeat_df(pos_train, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_test = df_test[df_test["label"] == 1]
        # pos_test = repeat_df(pos_test, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_val = df_val[df_val["label"] == 1]
        # pos_val = repeat_df(pos_val, self.repeat_pos_data, len(neg_df) // len(pos_df))

        neg_train = df_train[df_train["label"] == 0]
        neg_test = df_test[df_test["label"] == 0]
        neg_val = df_val[df_val["label"] == 0]

        train = np.array(pd.concat([pos_train, neg_train]).sample(frac=1)).tolist()
        test = np.array(pd.concat([pos_test, neg_test]).sample(frac=1)).tolist()
        val = np.array(pd.concat([pos_val, neg_val]).sample(frac=1)).tolist()

        save_json_data(self.train_data_path, train)
        save_json_data(self.test_data_path, test)
        save_json_data(self.validation_data_path, val)
        logging.info("split data done! all:{} train:{}({},{}) test:{}({},{}) val:{}({},{})".format(len(df),
                                                                                                   len(train),
                                                                                                   len(pos_train),
                                                                                                   len(neg_train),
                                                                                                   len(test),
                                                                                                   len(pos_test),
                                                                                                   len(neg_test),
                                                                                                   len(val),
                                                                                                   len(pos_val),
                                                                                                   len(neg_val)))

    def split_labeled_data_backup(self):
        all_data = read_json_data(self.label_data_path)
        import pandas as pd

        def repeat_df(df_data, repeat_pos_data_flag, neg_pos):
            if repeat_pos_data_flag >= 1:
                if repeat_pos_data_flag == 1:
                    # 自适应
                    if neg_pos > 1:
                        df_data = pd.concat([df_data] * neg_pos)
                else:
                    df_data = pd.concat([df_data] * neg_pos)
            return df_data

        df = pd.DataFrame(all_data, columns=["o_p", "kb_p", "label"])
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]
        # pos_df = repeat_df(pos_df, self.repeat_pos_data, len(neg_df) // len(pos_df))

        pos_df = pos_df.sample(frac=1)
        neg_df = neg_df.sample(frac=1)

        occupy = [0.6, 0.2, 0.2]
        train_num, test_num, val_num = int(occupy[0] * len(pos_df)), int(occupy[1] * len(pos_df)), int(
            occupy[2] * len(pos_df))
        pos_train = pos_df[:train_num]
        pos_train = repeat_df(pos_train, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_test = pos_df[train_num:train_num + test_num]
        pos_test = repeat_df(pos_test, self.repeat_pos_data, len(neg_df) // len(pos_df))
        pos_val = pos_df[train_num + test_num:]
        pos_val = repeat_df(pos_val, self.repeat_pos_data, len(neg_df) // len(pos_df))

        train_num, test_num, val_num = int(occupy[0] * len(neg_df)), int(occupy[1] * len(neg_df)), int(
            occupy[2] * len(neg_df))
        neg_train = neg_df[:train_num]
        neg_test = neg_df[train_num:train_num + test_num]
        neg_val = neg_df[train_num + test_num:]

        train = np.array(pd.concat([pos_train, neg_train]).sample(frac=1)).tolist()
        test = np.array(pd.concat([pos_test, neg_test]).sample(frac=1)).tolist()
        val = np.array(pd.concat([pos_val, neg_val]).sample(frac=1)).tolist()

        save_json_data(self.train_data_path, train)
        save_json_data(self.test_data_path, test)
        save_json_data(self.validation_data_path, val)
        logging.info("split data done! all:{} train:{} test:{} val:{}".format(len(df), len(train), len(test), len(val)))

    def control_node_num(self, graph, diy_node_num=None):
        """强行控制图的结点数为 diy_node_num """
        A_list = list()
        feature = graph[0]
        adj = graph[1]
        logging.info("process_graph_init: features:{} adj:{}".format(graph[0].shape, graph[1].shape))

        max_node_num = diy_node_num
        if max_node_num - feature.shape[0] < 0:
            """ 需要删除一些 """
            # 处理邻接矩阵
            num_delete = feature.shape[0] - max_node_num
            node_index_value = {_: graph[2][_] for _ in range(len(graph[2]))}
            for _ in graph[2]:
                # assert not isinstance(_, str)
                if isinstance(_, str):
                    node_index_value = {_: 0 for _ in range(len(node_index_value))}
                    logging.warning("some value is str!")
                    break

            node_index_value_sort = sorted(node_index_value.items(), key=lambda x: x[1])
            indexs_delete = np.array([node_index_value_sort[_][0] for _ in range(num_delete)])
            # 处理特征矩阵
            feature_new = np.delete(feature, indexs_delete, axis=0)
            # 处理邻接矩阵
            adj_r_array = adj
            adj_r_array = np.delete(adj_r_array, indexs_delete, axis=0)
            adj_r_array = np.delete(adj_r_array, indexs_delete, axis=1)
            # 处理node_value
            node_value = np.delete(graph[2], indexs_delete, axis=0)
            assert feature_new.shape[0] == adj_r_array.shape[0] == adj_r_array.shape[1] == node_value.shape[0]
            return feature_new, adj_r_array, node_value
        else:
            return graph[0], graph[1], graph[2]

    def process_graph(self, graph):
        """
        将一个feature矩阵和邻接矩阵 padding 后 转为 [feature, r_r, r_reverse, r_self]
        :param graph: (feature矩阵:np.array， 邻接稀疏矩阵: sp.csr_matrix, index_关键性：list)
        :param max_node_num:
        :return:
        """
        A_list = list()
        feature = graph[0]
        adj = graph[1]
        logging.info("process_graph_init: features:{} adj:{}".format(graph[0].shape, graph[1].shape))

        max_node_num = self.max_node_num
        if max_node_num - feature.shape[0] > 0:
            """ 需要 padding"""
            padding_num = max_node_num - feature.shape[0]
            # 处理特征
            feature_new = np.concatenate([feature, np.zeros((padding_num, feature.shape[1]))], axis=0)
            adj_new = np.concatenate([adj, np.zeros((padding_num, adj.shape[1]))], axis=0)
            adj_r = np.concatenate([adj_new, np.zeros((adj_new.shape[0], padding_num))], axis=1)
            adj_l = np.transpose(adj_r)

        elif max_node_num - feature.shape[0] < 0:
            """ 需要删除一些 """
            # 处理邻接矩阵
            num_delete = feature.shape[0] - max_node_num
            node_index_value = {_: graph[2][_] for _ in range(len(graph[2]))}
            for _ in graph[2]:
                # assert not isinstance(_, str)
                if isinstance(_, str):
                    node_index_value = {_: 0 for _ in range(len(node_index_value))}
                    logging.warning("some value is str!")
                    break

            node_index_value_sort = sorted(node_index_value.items(), key=lambda x: x[1])
            indexs_delete = np.array([node_index_value_sort[_][0] for _ in range(num_delete)])
            # 处理特征矩阵
            feature_new = np.delete(feature, indexs_delete, axis=0)
            # 处理邻接矩阵
            adj_r_array = adj
            adj_r_array = np.delete(adj_r_array, indexs_delete, axis=0)
            adj_r_array = np.delete(adj_r_array, indexs_delete, axis=1)

            adj_r = adj_r_array
            adj_l = np.transpose(adj_r)
        else:
            """形状不需要改变"""
            feature_new = feature
            adj_r = adj
            adj_l = np.transpose(adj_r)

        A_list.append(adj_r)
        A_list.append(adj_l)
        A_list.append(np.identity(adj_r.shape[0]))
        logging.info(
            "process_graph_done: features:{} adj_r:{} adj_l:{} self:{}".format(feature_new.shape, adj_r.shape,
                                                                               adj_l.shape, A_list[2].shape))
        return feature_new, np.array(A_list)


def read_json_data(read_path):
    with open(read_path, 'r', encoding='utf-8') as file_reader:
        raw_data = file_reader.read()
        paths_list = json.loads(raw_data)
    return paths_list


def read_pickle_data(read_path):
    with open(read_path, 'rb') as file_reader:
        return pickle.load(file_reader)


def save_json_data(save_path, pre_save_data):
    with open(save_path, 'w', encoding='utf-8') as file_writer:
        raw_data = json.dumps(pre_save_data, indent=4)
        file_writer.write(raw_data)


def save_pickle_data(save_path, pre_save_data):
    with open(save_path, 'wb') as f:
        pickle.dump(pre_save_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # 由当前线上与线下数据关系，生成标注数据,如将kb_construction/sock_shop 于 kb_online_construction/sockshop 进行整合,
    # 存储再config中标明的datasetversion文件夹里
    # 读取config
    # config_ds_ger = configparser.ConfigParser()
    # config_file_path = os.path.join(os.path.dirname(__file__), "config_graph_sim.ini")
    # config_ds_ger.read(config_file_path, encoding='utf-8')
    # dataset_version = config_ds_ger.get("data", "dataset_version")
    #
    # dsg = DataSetGraphSimGenerator(data_set_id=1, dataset_version="")
    # dsg.generate_dataset_pickle()
    ds = CustomDataset(data_set_id=1, dataset_version="", max_node_num=100, mode="val", repeat_pos_data=1)
    # sample = ds.graph_class_data()
    # ds.split_labeled_data()
    # ds.random_expand_train_data(expand_num=9,delete_part=0.1)
    # ds.guocaiyang_train()
    # pass


    # ds.generate_dataset_pickle_test()
    # for _ in ds.get_graph_pairs():
    #     pass

    """
    1、能否转为tensor形式使用 tensor
    2、能否用embedding 直接获取特征举证
    """

