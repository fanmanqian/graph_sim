# 构建知识库
import json
import logging
import os
import time
import sys
from collections import OrderedDict
from pprint import pprint
sys.path.append("/home/mfm/experiment")
import numpy as np
from KnowledgeBase import KnowledgeBase
from events_get.events import Events
#from generalConfig import generalConfig
from kb_algorithm.relation_classifier.event_vectorization import VectorizeEventPair
from kb_algorithm.relation_classifier.svm_model import SVM
import subprocess

class KBConstruction:
    """构建某种故障的知识库"""

    def __init__(self, data_set_id, fault_scenario):
        self.data_set_id = data_set_id
        self.data_set_name = self.__get_date_set_name()
        self.offline_data_set_info = self.__get_offline_data_set_info()
        # self.fault_scenarios = self.__get_fault_scenarios()
        self.fault_scenario = fault_scenario
        self.kb = KnowledgeBase()
        pass

    def __get_date_set_name(self):
        data_set_name = ""
        if self.data_set_id == 1:
            data_set_name = "train_ticket"
        elif self.data_set_id == 2:
            data_set_name = "sock_shop"
        return data_set_name

    @staticmethod
    def __get_offline_data_set_info():
        offline_data_set_info_path = os.path.join("/home/mfm/experiment/kb_algorithm/data/data_set_info/offline_data_set_info.json")
        offline_data_set_info = read_json_data(offline_data_set_info_path)
        return offline_data_set_info

    def __get_fault_scenarios(self):
        fault_scenarios = self.offline_data_set_info[self.data_set_name].keys()
        return list(fault_scenarios)[:-1]

    def __get_severity_threshold(self):
        severity_threshold = {
            'train_ticket': 
{'Execution': 0.7, 'Row Lock Contention': 0.7, 'LogFile': 0.7, 'DBFile;LogFile': 0.7, 'Index Contention': 0.7, 'Global Cache Buffer Busy': 0.7, 'LogFile;DBFile': 0.7, 'Contention': 0.7, 'Memory': 0.7, 'Execution;Execution': 0.7, 'Execution;Execution;Contention': 0.7, 'ASM Control File': 0.7, 'Parse': 0.7, 'CacheLock': 0.7, 'Global Cache Block': 0.7, 'Index Contention;Execution': 0.7, 'Execution;DBFile;DBFile': 0.7, 'Execution;Index Contention;LogFile': 0.7, 'DBFile;DBFile;DBFile': 0.7, 'Disk': 0.7, 'Contention;Row Lock Contention': 0.7, 'Global Cache Buffer Busy;LogFile': 0.7, 'Contention;LogFile': 0.7}
            ,
            'sock_shop': {
                'k_catalogue_db_catalogue_goods_disappeared': 0.6,
                'k_cart_db_cart_cart_disappeared': 0.85,
                'k_user_db_user_unable_log_in': 0.8,
                'k_order_db_order_order_disappeared': 0.8,
                'order_cart_500': 0.6,
                'order_count_500': 0.6,
                'order_payment_500': 0.7,
                'user_user_register_500': 0.7,
                'cart_cart_cart_disappeared': 0.7,
                'catalogue_db_catalogue_goods_disappeared': 0.7,
                'net_delay_cart_db_cart_add_cart_delay': 0.7,
                'net_delay_user_db_user_register_and_log_in_delay': 0.6,
                'net_delay_order_db_order_check_order_delay': 0.8,
                'net_loss_order_db_order_check_order_delay': 0.8,
                'net_loss_cart_db_cart_add_cart_delay': 0.7,
                'net_loss_user_db_user_register_and_log_in_delay': 0.7,
                'net_loss_catalogue_db_catalogue_goods_appear_delay': 0.7,
            },
        }
        return severity_threshold[self.data_set_name]

    def __judge_relation(self, start_node, end_node, time_window):
        """
        判别两个结点间，是否有关系，应该是什么关系
        :param start_node: dict()
        :param end_node: dict()
        :param time_window: 只保留指定时间窗口内可能存在的关系, 单位s
        :return:
        """
        if start_node['id'] == end_node['id']:
            return None, "same_event_diff_time"
        end_node_ts = end_node['time_stamp']
        start_node_ts = start_node['time_stamp']
        assert isinstance(end_node_ts, int) and isinstance(start_node_ts, int)
        time_gap = end_node_ts - start_node_ts
        time_gap = time_gap if time_gap > 0 else -time_gap
        # 超过时间窗口则无关系
        if time_gap > time_window:
            return None, "time_out_gap{}_win{}_ss{}_es{}".format(time_gap, time_window, start_node_ts, end_node_ts)
        # 同一时间同一设备上为conexist关系
        if start_node_ts == end_node_ts and start_node["location"] == end_node["location"]:
            return None, "same_device_same_time"
        # 剩下均可视作cause关系
        return 'cause', "cause"

    def get_candidate_relations(self, event_list, time_window=300):
        event_node_sorted = sorted(event_list, key=lambda _: _['time_stamp'])
        event_node_num = len(event_node_sorted)
        all_relation = list()
        # 初步生成所有的relation 即  n(n-1)/2 条关系
        logging.warning("{}{}{}".format("*" * 50, " init_nodes_relations ", "*" * 50))
        relation_beyond_time_num = 0
        for start_index in range(event_node_num - 1):
            end_index = start_index + 1
            while end_index < event_node_num:
                start_node = event_node_sorted[start_index]
                end_node = event_node_sorted[end_index]
                relation_name, _ = self.__judge_relation(start_node, end_node, time_window=time_window)
                if relation_name:
                    all_relation.append((start_node, relation_name, end_node))
                else:
                    relation_beyond_time_num += 1
                    # logging.info("init_no_relation : {} between {}--{}".format(_, start_node["id"], end_node["id"]))
                end_index += 1
        return all_relation

    def process_data(self, fault_scenario, critical_save_node_num, use_online_classifier, fault_scenario_num=-1,
                     reget=True):
        """对此故障场景中的事件进行预处理，删除大部分无用事件（筛选方法：1、关键性 2、沉默周期）"""
        s_path = os.path.join(os.path.dirname(__file__), "event_graph_data", self.data_set_name, self.fault_scenario)
        os.makedirs(s_path, exist_ok=True)
        s_path = os.path.join(s_path, "processed_data.json")
        if not reget and os.path.exists(s_path):
            processed_data = read_json_data(s_path)
            return processed_data
        processed_data = []
        fault_severity_threshold = self.__get_severity_threshold()[fault_scenario]
        fault_infos = self.offline_data_set_info[self.data_set_name][fault_scenario]
        if fault_scenario_num == -1:
            choosed_infos = fault_infos[:]
        else:
            choosed_infos = fault_infos[:fault_scenario_num]
        for fault_info in choosed_infos:
            # 远程拉取更新后的设备图
            # kg = self.kb
            # try:
            #     kg.download_graph_from_logstore(graphType="device", time_stamp=fault_info['device_graph_time_stamp'])
            #     logging.info("download device graph finished {}".format(fault_info['device_graph_time_stamp']))
            # except Exception:
            #     logging.error("download device graph failed!{}".format(fault_info['device_graph_time_stamp']))
            #     continue
            # 设置筛选阈值
            #generalConfig.lay_screen_2_threshold = fault_severity_threshold
            # 获取筛选后的事件
            event = Events()
            if use_online_classifier:
                event_list = event.get_all_events(
                    start_time=time_stamp_to_date(fault_info['start_time_stamp']),
                    end_time=time_stamp_to_date(fault_info['end_time_stamp']),
                    is_tf_idf=True,
                    is_alarm_sil=True,
                    is_period_del=False,
                    scene_type=None,
                    alarm_sil_period=10,
                    period_del_threshold=5,
                    period_del_protect=3,
                    data_set=self.data_set_id,
                    cri_save_node_num=critical_save_node_num
                )
            else:
                event_list = event.get_all_events(
                    start_time=time_stamp_to_date(fault_info['start_time_stamp']),
                    end_time=time_stamp_to_date(fault_info['end_time_stamp']),
                    is_tf_idf=True,
                    is_alarm_sil=True,
                    is_period_del=False,
                    scene_type=fault_scenario,
                    alarm_sil_period=10,
                    period_del_threshold=5,
                    period_del_protect=3,
                    data_set=self.data_set_id,
                    cri_save_node_num=critical_save_node_num
                )
            processed_data.append((event_list, fault_info['device_graph_time_stamp']))
        save_data(s_path, processed_data)
        return processed_data

    def get_construction_relations(self, critical_save_node_num, use_online_classifier, reget_process_data,
                                   use_peroid_num_per_scene, rebuild=True, combination_strategy=2):
        """生成构造此故障场景事件图所需要的所有关系"""
        # 当知识库中已经存在该事件图时，直接拉取使用
        kb_dir = os.path.join(os.path.dirname(__file__), "event_graph_data", self.data_set_name, self.fault_scenario)
        info_file = os.path.join(kb_dir, "nodes_relations.json")
        logging.info("rebuild: {} info_file: {}{} decide:{}".format(rebuild, info_file, os.path.exists(info_file),
                                                                    (not rebuild) and os.path.exists(info_file)))
        if (not rebuild) and os.path.exists(info_file):
            nodes_relations_json = read_json_data(os.path.join(kb_dir, "nodes_relations.json"))
            # 整理结点信息
            nodes_dict = nodes_relations_json["nodes_dict"]
            for _ in nodes_dict:
                _["event_type"] = _["_label"]
                _["detail_info"] = _["detail"]
            id_nodes = {node["id"]: node for node in nodes_dict}
            #  整理边的信息
            relations_dict = nodes_relations_json["relations_dict"]
            relation_list = list()
            for _r in relations_dict:
                relation_list.append((id_nodes[_r["_start_node_id"]], "cause", id_nodes[_r["_end_node_id"]]))
            node_list = nodes_dict
            logging.warning("kb {} - {} existed!  {}".format(self.data_set_name, self.fault_scenario, info_file))
            return relation_list, node_list

        # 知识库不存在，则重新构建
        logging.warning("build kb {} - {}".format(self.data_set_name, self.fault_scenario))
        relation_of_per_scene_list = []
        # 获取经过数据处理后的事件 [(event_list, timestamp), (list, int)]
        processed_data = self.process_data(
            self.fault_scenario, critical_save_node_num, fault_scenario_num=use_peroid_num_per_scene,
            use_online_classifier=use_online_classifier, reget=reget_process_data
        )
        processed_num = 0
        for _tuple in processed_data:
            processed_num += 1
            logging.warning(
                "{}/{} :: events-{} device_timestamp-{}".format(processed_num, len(processed_data), len(_tuple[0]),
                                                                _tuple[1]))
            event_list = _tuple[0]
            device_graph_time_stamp = _tuple[1]
            # 生成候选关系 [(start_node, relation_name, end_node), (dict, str, dict)]
            candidate_relations = self.get_candidate_relations(event_list)
            # 事件对 对应特征矩阵，计算每一对关系的 特征向量，与relation_list顺序一一对应
            if use_online_classifier:
                vep = VectorizeEventPair(self.kb, candidate_relations, event_list,
                                         device_graph_time_stamp=device_graph_time_stamp,
                                         data_set_type=self.data_set_name,
                                         error_type="")
            else:
                vep = VectorizeEventPair(self.kb, candidate_relations, event_list,
                                         device_graph_time_stamp=device_graph_time_stamp,
                                         data_set_type=self.data_set_name,
                                         error_type=self.fault_scenario)
            relation_pairs_values = vep.vectorize_event_pairs()
            # 去除第一列数据 将 period_1 去除
            relation_pairs_values = np.delete(relation_pairs_values, 0, axis=1)
            logging.info(
                "relation_attribute:{} relation_list:{}".format(relation_pairs_values.shape, len(candidate_relations)))
            assert relation_pairs_values.shape[0] == len(candidate_relations)

            # 使用SVM判别关系对
            if use_online_classifier:
                svm = SVM(dataset_id=self.data_set_id, offline=False)
            else:
                svm = SVM(dataset_id=self.data_set_id, offline=True)
            predict_values = svm.predict_by_batch(relation_pairs_values, batch_size=10000)
            logging.info("svm_predict_done: {}: {} - {}".format(len(predict_values), np.min(predict_values),
                                                                np.max(predict_values)))
            assert len(predict_values) == relation_pairs_values.shape[0]
            logging.info("predict_values:{}".format(predict_values.shape))
            pos_index = np.where(predict_values > 0)[0].tolist()
            new_relations = [candidate_relations[index] for index in pos_index]
            logging.warning("svm_judge: {}={}-{}".format(len(new_relations), len(candidate_relations),
                                                         len(candidate_relations) - len(new_relations)))
            relation_of_per_scene_list.append(new_relations)
        assist_list = []
        relation_list = []
        if combination_strategy == 1:
            # 获取所有节点id及其关键性
            node_id_cri_pairs = []
            for _r_list in relation_of_per_scene_list:
                for _r in _r_list:
                    node_id_cri_pair_0 = (_r[0]['id'], _r[0]['criticality'])
                    if node_id_cri_pair_0 not in node_id_cri_pairs:
                        node_id_cri_pairs.append(node_id_cri_pair_0)
                    node_id_cri_pair_1 = (_r[2]['id'], _r[2]['criticality'])
                    if node_id_cri_pair_1 not in node_id_cri_pairs:
                        node_id_cri_pairs.append(node_id_cri_pair_1)
            node_id_cri_pairs.sort(key=lambda _: _[1], reverse=True)
            node_id_cri_pairs = node_id_cri_pairs[:critical_save_node_num]
            _node_id_list = list(map(lambda _: _[0], node_id_cri_pairs))
            # 删除自环关系和节点关键性不够的关系
            for _r_l in relation_of_per_scene_list:
                for _r in _r_l:
                    # 去除节点关键性不足的关系
                    if _r[0]['id'] not in _node_id_list or _r[2]['id'] not in _node_id_list:
                        continue
                    # 去除自环关系
                    id_pair_set_1 = [_r[0]['id'], _r[2]['id']]
                    id_pair_set_2 = [_r[2]['id'], _r[0]['id']]
                    if id_pair_set_1 not in assist_list and id_pair_set_2 not in assist_list:
                        assist_list.append(id_pair_set_1)
                        relation_list.append(_r)
            pass
        elif combination_strategy == 2:
            # 获取所有节点id及其关键性
            node_id_rank_dict = OrderedDict()
            for _r_list in relation_of_per_scene_list:
                # 获取时间段所有节点id
                _node_id_list = []
                for _r in _r_list:
                    if _r[0]['id'] not in _node_id_list:
                        _node_id_list.append(_r[0]['id'])
                    if _r[2]['id'] not in _node_id_list:
                        _node_id_list.append(_r[2]['id'])
                for _node_id in _node_id_list:
                    if _node_id not in node_id_rank_dict:
                        node_id_rank_dict[_node_id] = 1
                    else:
                        node_id_rank_dict[_node_id] += 1
            node_id_rank_tuple = sorted(node_id_rank_dict.items(), key=lambda _: _[1], reverse=True)
            node_id_rank_tuple = node_id_rank_tuple[:critical_save_node_num]
            _node_id_list = list(map(lambda _: _[0], node_id_rank_tuple))
            # 删除自环关系和节点关键性不够的关系
            for _r_l in relation_of_per_scene_list:
                for _r in _r_l:
                    # 去除节点关键性不足的关系
                    if _r[0]['id'] not in _node_id_list or _r[2]['id'] not in _node_id_list:
                        continue
                    # 去除自环关系
                    id_pair_set_1 = [_r[0]['id'], _r[2]['id']]
                    id_pair_set_2 = [_r[2]['id'], _r[0]['id']]
                    if id_pair_set_1 not in assist_list and id_pair_set_2 not in assist_list:
                        assist_list.append(id_pair_set_1)
                        relation_list.append(_r)

            node_list1 = []
            for _r in relation_list:
                if _r[0] not in node_list1:
                    node_list1.append(_r[0])
                if _r[2] not in node_list1:
                    node_list1.append(_r[2])

            pass
        else:
            # 删除自环关系
            for _r_l in relation_of_per_scene_list:
                for _r in _r_l:
                    id_pair_set_1 = [_r[0]['id'], _r[2]['id']]
                    id_pair_set_2 = [_r[2]['id'], _r[0]['id']]
                    if id_pair_set_1 not in assist_list and id_pair_set_2 not in assist_list:
                        assist_list.append(id_pair_set_1)
                        relation_list.append(_r)
        # 获取节点列表
        node_list = []
        for _r in relation_list:
            if _r[0] not in node_list:
                node_list.append(_r[0])
            if _r[2] not in node_list:
                node_list.append(_r[2])
        event_node_sorted = sorted(node_list, key=lambda _: _['time_stamp'])
        return relation_list, event_node_sorted

    def construct_kb(self, reget_process_data, use_peroid_num_per_scene, rebuild=True, critical_save_node_num=200,
                     combination_strategy=2,
                     use_online_classifier=False):

        optimized_relations, event_node_sorted = self.get_construction_relations(
            rebuild=rebuild,
            use_peroid_num_per_scene=use_peroid_num_per_scene,
            reget_process_data=reget_process_data,
            combination_strategy=combination_strategy,
            critical_save_node_num=critical_save_node_num,
            use_online_classifier=use_online_classifier,
        )
        print("optimized_relations")
        print(optimized_relations)
        # for node in node_list:
        #     if not isinstance(node, dict):
        #         logging.warning("{} :: {}".format(type(node), node))

        values = np.ones((len(optimized_relations),))
        logging.info("construct_kb: {} relations: {}  nodes: {}".format(self.fault_scenario, len(optimized_relations),
                                                                        len(event_node_sorted)))
        construct_result, relations = self.kb.construct_event_graph_nodes_relations_together(
            node_list=event_node_sorted,
            relation_list=optimized_relations,
            relation_values=values.tolist(),
            node_label_key="event_type",
            node_id_key="id",
            node_location_key="location",
            startTime="",
            endTime="",
            relation_no_direction=["coexist"],
            printNode=False,
        )
        self.kb.save_event_graph2local(offline=True, data_set_type=self.data_set_name, fault_type=self.fault_scenario)
        save_data(
            os.path.join(os.path.dirname(__file__), "event_graph_data", self.data_set_name, self.fault_scenario,
                         "before_svm.json"),
            {
                "relations": optimized_relations,
                "nodes": event_node_sorted,
            }
        )

        assert construct_result
        pass


def read_json_data(read_path):
    with open(read_path, 'r', encoding='utf-8') as file_reader:
        raw_data = file_reader.read()
        paths_list = json.loads(raw_data)
        #print(paths_list)
    return paths_list


def save_data(save_path, pre_save_data):
    with open(save_path, 'w', encoding='utf-8') as file_writer:
        raw_data = json.dumps(pre_save_data, indent=4)
        file_writer.write(raw_data)


def date_to_time_stamp(date):
    """将不带Z的日期转换为时间戳"""
    a = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    a = time.mktime(a)
    a = int(a)
    return a


def time_stamp_to_date(time_stamp):
    a = time.localtime(time_stamp)
    a = time.strftime("%Y-%m-%d %H:%M:%S", a)
    return a


if __name__ == "__main__":
    offline_data_set_info = read_json_data("/home/mfm/experiment/kb_algorithm/data/data_set_info/offline_data_set_info.json")
    fault_scenarios = offline_data_set_info['train_ticket'].keys()
    f_all_1 = []
    for elem in fault_scenarios:
        f_all_1.append(elem)
    #print(f_all_1)

    # f_all_1 = ["f1", "f2", "f3", "f5", "f6", "f10", "f13", "f16", "f17", "f18"]
    # f_all_1 = ["f2", "f3", "f5", "f10", "f13", "f16", "f17", "f18"]
    f_all_2 = [
        # 'k_catalogue_db_catalogue_goods_disappeared',
        # 'k_cart_db_cart_cart_disappeared',
        'k_user_db_user_unable_log_in',
        # 'k_order_db_order_order_disappeared',
        # 'order_cart_500',
        # 'order_count_500',
        # 'order_payment_500',
        # 'user_user_register_500',
        # 'cart_cart_cart_disappeared',
        # 'catalogue_db_catalogue_goods_disappeared',
        # 'net_delay_cart_db_cart_add_cart_delay',
        # 'net_delay_user_db_user_register_and_log_in_delay',
        # 'net_delay_order_db_order_check_order_delay',
        # 'net_loss_order_db_order_check_order_delay',
        # 'net_loss_cart_db_cart_add_cart_delay',
        # 'net_loss_user_db_user_register_and_log_in_delay',
        # 'net_loss_catalogue_db_catalogue_goods_appear_delay',
    ]
    _data_set_id = 1
    f_all = f_all_1 if _data_set_id == 1 else f_all_2
    for f in f_all:
        if f == "normal":
            continue
        logging.warning("start {}".format(f))
        kb_cons = KBConstruction(data_set_id=_data_set_id, fault_scenario=f)
        # cmd_command = 'python /home/mfm/experiment/test.py'
        # result = subprocess.run(cmd_command, shell=True, capture_output=True, text=True)
        # relation_list, node_list = kb_cons.get_construction_relations()
        # print(len(relation_list))
        # print(len(node_list))
        kb_cons.construct_kb(
            rebuild=True, critical_save_node_num=10,
            combination_strategy=2,
            use_peroid_num_per_scene=-1,
            # 表示多个时间段组成一个场景事件图时节点保留的策略，1表示使用关键性排序保留前n个点，2表示使用节点的soft交集, 其它表示保留全部节点
            use_online_classifier=False, reget_process_data=False
        )
        # os.system("python ")
        cmd_command = 'python /home/mfm/experiment/test.py'
        result = subprocess.run(cmd_command, shell=True, capture_output=True, text=True)

    # _data_set_id = 2
    # f_all = f_all_1 if _data_set_id == 1 else f_all_2
    # for f in f_all:
    #     logging.warning("start {}".format(f))
    #     kb_cons = KBConstruction(data_set_id=_data_set_id, fault_scenario=f)
    #     # # relation_list, node_list = kb_cons.get_construction_relations()
    #     # # pprint(len(relation_list))
    #     # # pprint(len(node_list))
    #     kb_cons.construct_kb(
    #         rebuild=True, critical_save_node_num=100,
    #         combination_strategy=2,
    #         use_peroid_num_per_scene=-1,
    #         # 表示多个时间段组成一个场景事件图时节点保留的策略，1表示使用关键性排序保留前n个点，2表示使用节点的soft交集, 其它表示保留全部节点
    #         use_online_classifier=True, reget_process_data=True
    #     )

    # kb_cons = KBConstruction(data_set_id=1, fault_scenario="f18")
    # kb_cons.construct_kb()
