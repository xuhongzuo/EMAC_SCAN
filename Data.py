import pandas as pd
from collections import Counter
import numpy as np
import math


class Data:
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        [self.objects_num, self.features_num] = data_matrix.shape

        self.values_num_list = np.array([len(np.unique(self.data_matrix[:, i])) for i in range(self.features_num)])
        self.values_num = np.sum(self.values_num_list)

        self.cate_data = np.zeros([self.objects_num, self.features_num], dtype=int)

        self.mode_frequency = np.zeros(self.features_num, dtype=int)
        self.value_frequency_list = np.zeros(self.values_num)
        self.value_list = []
        self.first_value_index = []
        self.first_value_index.append(0)
        self.value_feature_indicator = []

        self.co_occurrence = np.zeros([self.values_num, self.values_num])
        self.conditional_possibility = np.zeros([self.values_num, self.values_num])
        self.similarity_matrix = np.zeros([self.values_num, self.values_num])

        return

    # calculate basic statistical information
    def data_prepare(self):
        # calc first_value_index, count value frequency,
        # generate value list for each feature, indicate the feature index of the values
        for i in range(self.features_num):
            column = self.data_matrix[:, i]
            this_value_list = np.unique(column).tolist()
            feature_value_num = len(this_value_list)
            self.first_value_index.append(self.first_value_index[i] + feature_value_num)
            for j in range(feature_value_num):
                self.value_feature_indicator.append(i)

            frequency_map = Counter(column)
            for jj, item in enumerate(this_value_list):
                frequency = frequency_map.get(item)
                self.value_frequency_list[self.first_value_index[i] + jj] = frequency
            self.value_list.append(this_value_list)
            self.mode_frequency[i] = max(frequency_map.values())

        # process categorical space
        for i in range(0, self.features_num):
            this_value_list = self.value_list[i]
            this_value_index_map = {}
            for j in range(len(this_value_list)):
                this_value_index_map[this_value_list[j]] = self.first_value_index[i] + j
            for k in range(self.objects_num):
                self.cate_data[k][i] = this_value_index_map[self.data_matrix[k][i]]

        # calc co-occurrence frequency and list of class
        for obj in self.cate_data:
            for f1 in range(self.features_num):
                for f2 in range(f1, self.features_num):
                    self.co_occurrence[obj[f1]][obj[f2]] += 1
                    self.co_occurrence[obj[f2]][obj[f1]] = self.co_occurrence[obj[f1]][obj[f2]]

        # calculate conditional probability and ochiai similarity
        for i in range(self.values_num):
            i_frequency = self.value_frequency_list[i]
            for j in range(self.values_num):
                j_frequency = self.value_frequency_list[j]
                self.conditional_possibility[i][j] = float(self.co_occurrence[i][j]) / float(i_frequency)
                self.similarity_matrix[i][j] = float(self.co_occurrence[i][j]) / float(math.sqrt(i_frequency * j_frequency))

        return
