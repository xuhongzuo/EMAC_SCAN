from Data import Data
import GraphUtils
import numpy as np
import pandas as pd
import Utils
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score
from collections import Counter
from gensim.models import Word2Vec
import operator
import time


class SCAN:
    def __init__(self):
        # data
        self.data = None

        # parameters
        self.dimensions = 0
        self.walk_length = 0
        self.num_walks = 0
        self.window_size = 0
        self.alpha = 0

        # intermediate result
        self.primary_direct_coup_M1 = None
        self.primary_direct_coup_M2 = None
        self.primary_indirect_coup = None

        self.value_embeddings = None
        self.value_similarity = None
        self.value_final_score = None
        self.obj_score = None

        return

    def fit(self, data_matrix, dimensions=128, walk_length=80, num_walks=30, alpha=0.15):
        self.data = Data(data_matrix)
        self.data.data_prepare()

        # parameters
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = self.data.features_num
        self.alpha = alpha

        # OD
        self.calc_primary_coupling()
        init_score_eta = self.calc_init_score()
        adjacency_matrix, graph = self.construct_network(init_score_eta)
        self.value_embeddings, self.value_similarity = self.learn_embeddings(graph, adjacency_matrix)
        self.value_final_score = self.value_scoring(self.value_similarity, init_score_eta)
        self.obj_score = self.obj_scoring(self.value_final_score)
        return

    def calc_primary_coupling(self):
        values_num = self.data.values_num
        c_p = self.data.conditional_possibility

        self.primary_direct_coup_M1 = self.data.similarity_matrix
        self.primary_direct_coup_M2 = self.data.conditional_possibility

        # calculate indirect basic couplings between values
        self.primary_indirect_coup = np.zeros([values_num, values_num])
        for i in range(values_num):
            vector1 = c_p[i]
            for j in range(i + 1, values_num):
                vector2 = np.reshape(c_p[j], [values_num, 1])
                dot = np.dot(vector1, vector2)
                denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                self.primary_indirect_coup[i][j] = dot / denom
                self.primary_indirect_coup[j][i] = dot / denom

    def calc_init_score(self):
        values_num = self.data.values_num
        f_indicator = self.data.value_feature_indicator
        v_f_list = self.data.value_frequency_list
        mode_f = self.data.mode_frequency
        similarity_matrix = self.data.similarity_matrix

        # rough scoring function \delta(v) to obtain value subset
        init_value_outlierness = np.zeros(values_num)
        for i in range(values_num):
            init_value_outlierness[i] = float(mode_f[f_indicator[i]] - v_f_list[i]) / float(mode_f[f_indicator[i]])
        size = int(values_num * self.alpha)
        sorted_index = Utils.get_sorted_index(init_value_outlierness)
        top_value = sorted_index[:size]
        bottom_value = sorted_index[values_num-size:]

        # calculate initial value outlierness
        init_score_eta = np.zeros(values_num)
        for i in range(values_num):
            init_score_eta[i] = (np.sum(similarity_matrix[i][top_value]) +
                          np.sum(1 - similarity_matrix[i][bottom_value])) / float(2 * size)

        return init_score_eta

    def construct_network(self, init_score_eta):
        values_num = self.data.values_num

        # use spectral clustering to model complex couplings
        cluster_info_list = []
        k = 2
        while True:
            spectral = SpectralClustering(n_clusters=k, gamma=0.1, assign_labels="discretize").\
                fit(self.primary_direct_coup_M2)
            cluster_info = spectral.labels_
            con = Counter(cluster_info)
            tiny_cluster_num = np.sum(list(map(lambda x: x == 1, con.values())))
            if tiny_cluster_num > 0:
                break
            k += 1
            cluster_info_list.append(cluster_info)

        cluster_result = np.zeros([values_num, values_num])
        for cluster_info in cluster_info_list:
            for a in range(values_num):
                cluster_a = cluster_info[a]
                for b in range(values_num):
                    cluster_b = cluster_info[b]
                    if a != b and cluster_a == cluster_b:
                        cluster_result[a][b] += 1.
        cluster_result = cluster_result / len(cluster_info_list)

        # non-zero value coupling bias matrix
        bias = np.zeros([values_num, values_num])
        for i in range(values_num):
            for j in range(i, values_num):
                bias[i][j] = (1 + cluster_result[i][j]) * (1 + (init_score_eta[i] + init_score_eta[j]) * 0.5)
                bias[j][i] = bias[i][j]

        adjacency_matrix = np.zeros([values_num, values_num])
        for i in range(values_num):
            for j in range(i, values_num):
                adjacency_matrix[i][j] = self.primary_direct_coup_M1[i][j] * \
                                         self.primary_indirect_coup[i][j] * bias[i][j]
                adjacency_matrix[j][i] = adjacency_matrix[i][j]
        graph = Utils.generate_graph(adjacency_matrix)
        return adjacency_matrix, graph

    def learn_embeddings(self, graph, adjacency_matrix):
        values_num = self.data.values_num
        G = GraphUtils.Graph(graph, adjacency_matrix)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(self.num_walks, self.walk_length)
        model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=8, iter=5)

        value_embeddings = np.zeros([values_num, self.dimensions])
        for i in range(values_num):
            vector = model.wv[str(i)]
            value_embeddings[i] = vector

        value_similarity = np.zeros([values_num, values_num])
        for i in range(values_num):
            for j in range(i, values_num):
                value_similarity[i][j] = model.wv.similarity(str(i), str(j))
                value_similarity[j][i] = value_similarity[i][j]
        return value_embeddings, value_similarity

    def value_scoring(self, value_coupling, init_score_eta):
        values_num = self.data.values_num
        last_rank = Utils.get_sorted_index(init_score_eta)
        size = int(values_num * self.alpha)

        iter_count = 1
        while True:
            value_score = np.zeros(values_num)
            outlier_values = last_rank[:size]
            normal_values = last_rank[values_num - size:]
            for i in range(values_num):
                value_score[i] = (np.sum(value_coupling[i][outlier_values]) +
                                     np.sum(1 - value_coupling[i][normal_values])) / float(2 * size)
            rank = Utils.get_sorted_index(value_score)
            iter_count += 1
            if operator.eq(rank, last_rank):
                break
            else:
                last_rank = rank

        return value_score

    def obj_scoring(self, value_score):
        obj_score = np.array([np.sum(value_score[obj]) for obj in self.data.cate_data])
        return obj_score
