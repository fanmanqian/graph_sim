数据集：
Dej的数据集下载地址：
https://www.dropbox.com/sh/ist4ojr03e2oeuw/AAD5NkpAFg1nOI2Ttug3h2qja?dl=0
利用cal_faults.py，将Dej的D数据集下的graphs转化为本论文的最原始输入events_initial，
读取D数据集下的faults.csv构造本论文需要的offline_data_set_info.json和online_data_set_info.json。进行环境配置，需要配置log4j运行环境，然后才能执行下述流程，然后利用本论文对数据集的预处理方式进行train，test，val的划分：
1.利用KBConstruction.py构建故障图谱
2.利用DataSetGraphSimGenerator.py进行训练测试验证的划分

代码运行：
graph_sim_dej_X.py----------------本文提出的方法，其中X为预处理后的D和C数据集
graph_sim_no_gcn_dej_X.py --------本文提出的方法，不使用gcn的，其中X为预处理后的D和C数据集
graph_sim_no_kb_dej_X.py----------本文提出的方法，不使用kb的，其中X为预处理后的D和C数据集


