import numpy as np
import networkx as nx
from sklearn import metrics


def get_sorted_index(score, order="descending"):
    '''
    :param score:
    :param order:
    :return: index of sorted item in descending order
    e.g. [8,3,4,9] return [3,0,2,1]
    '''
    score_map = []
    size = len(score)
    for i in range(size):
        score_map.append({'index':i, 'score':score[i]})
    if order == "descending":
        reverse = True
    elif order == "ascending":
        reverse = False
    score_map.sort(key=lambda x: x['score'], reverse=reverse)
    keys = [x['index'] for x in score_map]
    return keys


# Reads the input network in networkx.
def generate_graph(adjacency_matrix):
    nValues = len(adjacency_matrix)
    G = nx.Graph()
    for i in range(nValues):
        for j in range(i + 1, nValues):
            if adjacency_matrix[i][j] != 0:
                node1 = str(i)
                node2 = str(j)
                weight = adjacency_matrix[i][j]
                G.add_edge(node1, node2, weight=weight)
    return G


def evaluation(score, y_true):
    auc_roc = metrics.roc_auc_score(y_true, score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, score)
    auc_pr = metrics.auc(recall, precision)
    return auc_roc, auc_pr
