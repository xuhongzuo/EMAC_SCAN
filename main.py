# @Time : 2019/3/19 21:15
# @Author : Hongzuo Xu
# @Description ：

import os
import warnings
import numpy as np
import pandas as pd
import time
from Utils import evaluation
from SCAN import SCAN


def main(path, runs=10):
    dataset_name = path.split("/")[-1].split(".")[0]

    x = pd.read_csv(input_path).values[:, :-1]
    y = pd.read_csv(input_path).values[:, -1]
    scan = SCAN()

    auroc_list = np.zeros(runs)
    aupr_list = np.zeros(runs)
    time_list = np.zeros(runs)
    for i in range(runs):
        time1 = time.time()
        scan.fit(data_matrix=x, dimensions=128, walk_length=80, num_walks=30, alpha=0.15)
        y_true = np.array(y, dtype=int)
        score = scan.obj_score
        time2 = time.time()
        auroc_list[i], aupr_list[i] = evaluation(score, y_true)
        time_list[i] = time2 - time1

    # summary = dataset_name + ", AUC-ROC, %.4f, %.4f , AUC-PR, %.4f, %.4f, %.4fs" % \
    #           (np.average(auroc_list), np.std(auroc_list), np.average(aupr_list),
    #            np.std(aupr_list), np.average(time_list))
    summary = dataset_name + ", AUC-ROC, %.4f, %.4f %.4fs" % \
              (np.average(auroc_list), np.std(auroc_list), np.average(time_list))
    doc = open('out.txt', 'a')
    print(summary, file=doc)
    doc.close()
    print(summary)

    return


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    input_root = 'data/05-solar.csv'
    run_times = 1
    if os.path.isdir(input_root):
        for file_name in os.listdir(input_root):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_root, file_name)
                main(input_path, runs=run_times)
    else:
        input_path = input_root
        main(input_path, runs=run_times)

