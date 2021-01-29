# %%
from lsopt.tree import OptimalTreeClassifier
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import datasets
import graphviz

# %%
current_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(current_path, "./data/Iris.csv")

iris_data = pd.read_csv(data_path)
iris_data.drop(columns=["Id"], inplace=True)
iris_data_sub = iris_data.sample(n=20, random_state=1)

# get X from iris data_sub
# iris_sub_X = iris_data_sub.iloc[:, 0:4].to_numpy()
# iris_sub_Y_label = iris_data_sub["Species"].to_numpy()

iris_all_X = iris_data.iloc[:, 0:4].to_numpy()
iris_all_Y_label = iris_data["Species"].to_numpy()

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# %%
# Define OptimalTreeClassifier
opt_tree = OptimalTreeClassifier(max_depth=3,
                                 min_samples_leaf=1,
                                 alpha=0.01,
                                 criterion="gini",
                                 solver="gurobi",
                                 time_limit=1,
                                 verbose=True,
                                 solver_options={'mip_cuts': 'auto',
                                                 'mip_gap_tol': 0.8,
                                                 'mip_focus': 'balance'}
                                 )

# # Fit on X and y
# opt_tree.fit(X=X, y=y)

# # Make prediction
# y_pred = opt_tree.predict(X=X)
# y_pred_prob = opt_tree.predict_proba(X=X)

# # Check confusion matrix
# print("Confusion Matrix :")
# print(confusion_matrix(y_true=y,
#                        y_pred=y_pred))

# print(classification_report(y_true=y,
#                             y_pred=y_pred))


# Fit on X and y
opt_tree.fit(X=iris_all_X, y=iris_all_Y_label)

# Make prediction
Y_pred = opt_tree.predict(X=iris_all_X)
Y_pred_prob = opt_tree.predict_proba(X=iris_all_X)

# Check confusion matrix
print("Confusion Matrix :")
print(confusion_matrix(y_true=iris_all_Y_label,
                       y_pred=Y_pred))

print(classification_report(y_true=iris_all_Y_label,
                            y_pred=Y_pred))

# Plot Optimal Tree
feature_names = iris_data.columns.values[:4]
class_names = np.unique(iris_all_Y_label)


dot_data = tree.export_graphviz(opt_tree,
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
graph.render(filename='optimal_tree', directory=current_path, view=True)

# add
