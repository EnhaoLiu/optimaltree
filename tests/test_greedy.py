# %%
from __future__ import division
import pyomo.common
from pyomo.common import fileutils
import pyomo
import sys
import cplex
import numpy as np
import pandas as pd
import math
import warnings
from sklearn import tree, datasets
import graphviz
import os
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# This script includes several main functions
# that are used to fit the Optimal Classification Trees based on MILP

# %%
current_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(current_path, "./data/Iris.csv")

iris_data = pd.read_csv(data_path)
iris_data.drop(columns=["Id"], inplace=True)
iris_data_sub = iris_data.sample(n=20, random_state=1)

# get X from iris data_sub
iris_sub_X = iris_data_sub.iloc[:, 0:4].to_numpy()
iris_sub_Y_label = iris_data_sub["Species"].to_numpy()

iris_all_X = iris_data.iloc[:, 0:4].to_numpy()
iris_all_Y_label = iris_data["Species"].to_numpy()
# %%
# Decision Tree
# set the maximum depth of the tree is 2

# tree_clf = tree.DecisionTreeClassifier(max_depth=3)
# tree_clf.fit(iris_sub_X, iris_sub_Y_label)
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

tree_clf = tree.DecisionTreeClassifier(max_depth=3, criterion="gini")
tree_clf.fit(iris_all_X, iris_all_Y_label)
# tree_clf.fit(X, y)
y_pred = tree_clf.predict(iris_all_X)

# Check confusion matrix
print("Confusion Matrix :")
print(confusion_matrix(y_true=iris_all_Y_label,
                       y_pred=y_pred))

print(classification_report(y_true=iris_all_Y_label,
                            y_pred=y_pred))

# %%
# Visualize the trained Decision Tree
feature_names = iris_data_sub.columns.values[:4]
class_names = np.unique(iris_sub_Y_label)


dot_data = tree.export_graphviz(tree_clf,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=class_names,
                                label='all',
                                impurity=True,
                                node_ids=True,
                                filled=True,
                                rounded=True,
                                leaves_parallel=True,
                                special_characters=False)

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render(filename='greedy_tree', directory=current_path, view=True)


# # # %%
# # for attr in dir(tree_clf.tree_):
# #     if "__" not in attr:
# #         print(attr, getattr(tree_clf.tree_, attr))
# # %%
# # for path in os.environ['PATH'].split(os.pathsep):
# #     print(path)


# # %%
# tree.plot_tree(tree_clf,
#                feature_names=feature_names,
#                class_names=class_names,
#                label='all',
#                filled=True,
#                impurity=True,
#                node_ids=False,
#                rounded=True)

# %%
