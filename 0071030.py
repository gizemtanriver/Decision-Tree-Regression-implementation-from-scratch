"""
Gizem Tanriver
Student ID: 0071030
Homework 5
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('hw05/hw05_data_set.csv')

# Splitting training and test sets
X_train=dataset.iloc[0:150, 0].values
X_test=dataset.iloc[150:, 0].values
y_train = dataset.iloc[0:150, 1].values
y_test = dataset.iloc[150:, 1].values

# get length of train and test set
N_train = len(y_train)
N_test = len(y_test)


def regressionTree(P, draw):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_splits = {}

    # put all training instances into root node
    node_indices[0] = list(range(N_train))
    is_terminal[0] = False
    need_split[0] = True

    # learning algorithm
    while True:
        # find node index that need splitting
        split_nodes = [i for i,v in need_split.items() if v==True]
        # check whether we reached all terminal nodes
        if len(split_nodes) == 0: break

        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False

            # If a node has 𝑃 or fewer data points, convert this node into a terminal node and do not split
            if len(node_indices[split_node])<=P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False

                unique_values = np.unique(X_train[data_indices])
                split_positions = (unique_values[1:] + unique_values[:-1])/2
                split_scores = np.zeros(len(split_positions))

                for s in range(len(split_positions)):
                    left_indices = []
                    [left_indices.append(data_indices[tf]) for tf in range(len((X_train[data_indices] < split_positions[s]))) if (X_train[data_indices] < split_positions[s])[tf]]
                    right_indices = []
                    [right_indices.append(data_indices[tf]) for tf in range(len((X_train[data_indices] >= split_positions[s]))) if (X_train[data_indices] >= split_positions[s])[tf]]
                    split_scores[s] = (1/len(data_indices))*(np.sum(np.square(y_train[left_indices] - np.mean(y_train[left_indices]))) + np.sum(np.square(y_train[right_indices] - np.mean(y_train[right_indices]))))

                # best_score = min(split_scores)
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split

                # create the left node using the selected split
                left_indices = []
                [left_indices.append(data_indices[tf]) for tf in range(len((X_train[data_indices] < best_split))) if (X_train[data_indices] < best_split)[tf] ]
                node_indices[2*split_node + 1] = left_indices
                is_terminal[2*split_node + 1] = False
                need_split[2*split_node + 1] = True

                # create the right node using the selected split
                right_indices = []
                [right_indices.append(data_indices[tf]) for tf in range(len((X_train[data_indices] >= best_split))) if (X_train[data_indices] >= best_split)[tf]]
                node_indices[2*split_node + 2] = right_indices
                is_terminal[2*split_node + 2] = False
                need_split[2*split_node +2] = True

    if draw==True:
    # Visualizing the regression tree
        dataInterval = np.arange(1.5, 5.2, 0.001)
        y_interval = np.zeros(len(dataInterval))
        for i in range(len(dataInterval)):
            index = 0
            while True:
                if is_terminal[index] == True:
                    y_interval[i] = np.mean(y_train[node_indices[index]])
                    break
                else:
                    if dataInterval[i] < node_splits[index]:
                        index = index * 2 + 1
                    else:
                        index = index * 2 + 2

        plt.scatter(X_train, y_train, color='blue', label="training")
        plt.scatter(X_test, y_test, color='red', label="test")
        plt.title('P=%s' % P)
        plt.xlabel('Eruption time (min)')
        plt.ylabel('Waiting time to next eruption (min)')
        plt.legend(loc="upper left")
        for i in range(len(dataInterval) - 1):
            plt.plot([dataInterval[i], dataInterval[i + 1]], [y_interval[i], y_interval[i + 1]], color='black')


    # traverse tree for test data points
    y_predicted = np.zeros(N_test)
    for i in range(N_test):
        index = 0
        while True:
            if is_terminal[index] == True:
                y_predicted[i] = np.mean(y_train[node_indices[index]])
                break
            else:
                if X_test[i] <= node_splits[index]: index = index * 2 + 1
                else: index = index * 2 + 2

    # Calculate the root mean squared error (RMSE) for test data points
    RMSE = round(math.sqrt(np.sum(np.square(y_test - y_predicted)) / len(y_test)), 4)
    print("Regressogram => RMSE is %s when P is %s" %(RMSE, P))
    return RMSE

# Regression Tree when P=25
regressionTree(25, draw=True)


# Decision Tree at varying P values
pList = list(range(5, 51, 5))
RMSEList = []
for p in pList:
    RMSEList.append(regressionTree(p, draw=False))
# Visualize RMSE for test data points as a function of 𝑃.
plt.plot(pList, RMSEList, 'b-o')
plt.xlabel('Pre−pruning size (P)')
plt.ylabel('RMSE')
plt.show()