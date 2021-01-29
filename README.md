# Large-Scale Optimal Classification Tree

This is a **python** implementation for the Optimal Classification Trees (OCTs) proposed by Dimitris Bertsimas and Jack Dunn, and a new algorithm called Large-Scale OCTs developed by Theodore Allen and Enhao Liu.

The implementation of `lsopt` is compatible with `scikit-learn` machine learning mudules. See **User Guide** for more details. 


## User Guide

The implementation of `lsopt` is compatible with `scikit-learn` machine learning mudules. 

### Example of Iris

Download the iris data from Kaggle : https://www.kaggle.com/uciml/iris

```python
import pandas as pd
import numpy as np

from lsopt.tree import OptimalTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn import tree
# import graphviz

iris_data = pd.read_csv("your data path/Iris.csv")
iris_data.drop(columns=["Id"], inplace=True)
iris_all_X = iris_data.iloc[:, 0:4].to_numpy()
iris_all_Y_label = iris_data["Species"].to_numpy()

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
```

```
Academic license - for non-commercial use only
Read LP format model from file /var/folders/5y/drctn5r14mzfjfjnld2ybpc80000gn/T/tmp0qbdh7hp.pyomo.lp
Reading time = 0.02 seconds
x1315: 5067 rows, 1315 columns, 28910 nonzeros
Changed value of parameter TimeLimit to 60.0
   Prev: inf  Min: 0.0  Max: inf  Default: inf
Parameter Cuts unchanged
   Value: -1  Min: -1  Max: 3  Default: -1
Changed value of parameter MIPGap to 0.8
   Prev: 0.0001  Min: 0.0  Max: inf  Default: 0.0001
Parameter MIPFocus unchanged
   Value: 0  Min: 0  Max: 3  Default: 0
Gurobi Optimizer version 9.0.1 build v9.0.1rc0 (mac64)
Optimize a model with 5067 rows, 1315 columns and 28910 nonzeros
Model fingerprint: 0xa96d3d8d
Variable types: 48 continuous, 1267 integer (1267 binary)
Coefficient statistics:
  Matrix range     [2e-02, 2e+02]
  Objective range  [1e-02, 1e-02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 2e+02]
Presolve removed 1 rows and 1 columns
Presolve time: 0.03s
Presolved: 5066 rows, 1314 columns, 26541 nonzeros
Variable types: 7 continuous, 1307 integer (1267 binary)
Found heuristic solution: objective 1.0000000

Root relaxation: objective 0.000000e+00, 589 iterations, 0.03 seconds

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0    0.00000    0    3    1.00000    0.00000   100%     -    0s
     0     0    0.00000    0   71    1.00000    0.00000   100%     -    0s
     0     0    0.00000    0  405    1.00000    0.00000   100%     -    0s
     0     0    0.00000    0  148    1.00000    0.00000   100%     -    0s
     0     0    0.00000    0  142    1.00000    0.00000   100%     -    0s
H    0     0                       0.9900000    0.01000  99.0%     -    0s
     0     2    0.01000    0  139    0.99000    0.01000  99.0%     -    0s
H   34    40                       0.5200000    0.01000  98.1%   119    0s
H   82    91                       0.4300000    0.01000  97.7%  70.3    1s
H   92    91                       0.2500000    0.01000  96.0%  63.3    1s
H  256   248                       0.1800000    0.01000  94.4%  31.9    1s
H  260   248                       0.1400000    0.01000  92.9%  31.5    1s
  1536  1154    0.05776   22  152    0.14000    0.02000  85.7%  34.5    5s
* 2327  1267             123       0.0800000    0.02000  75.0%  56.8    7s
H 2331  1211                       0.0600000    0.02000  66.7%  56.7    7s

Cutting planes:
  Learned: 7
  Cover: 400
  Implied bound: 4
  Clique: 26
  MIR: 1

Explored 2394 nodes (142453 simplex iterations) in 7.63 seconds
Thread count was 16 (of 16 available processors)

Solution count 9: 0.06 0.08 0.14 ... 1

Optimal solution found (tolerance 8.00e-01)
Best objective 6.000000000000e-02, best bound 2.000000000000e-02, gap 66.6667%
Solver running time: 7.747809886932373
Solver termination condition: optimal
Valid Tree : Yes
Confusion Matrix :
[[50  0  0]
 [ 0 47  3]
 [ 0  0 50]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        50
Iris-versicolor       1.00      0.94      0.97        50
 Iris-virginica       0.94      1.00      0.97        50

       accuracy                           0.98       150
      macro avg       0.98      0.98      0.98       150
   weighted avg       0.98      0.98      0.98       150
```

## classifier.tree_

- `apply`: <built-in method apply of sklearn.tree._tree.Tree object at 0x13a541d98>
- `compute_feature_importances`: <built-in method compute_feature_importances of sklearn.tree._tree.Tree object at 0x13a541d98>
- `compute_partial_dependence` : <built-in method compute_partial_dependence of sklearn.tree._tree.Tree object at 0x13a541d98>
- `decision_path`: <built-in method decision_path of sklearn.tree._tree.Tree object at 0x13a541d98>
- `predict` : <built-in method predict of sklearn.tree._tree.Tree object at 0x13a541d98>

- `capacity` : 5

- `children_left` : [ 1 -1  3 -1 -1]

- `children_right` : [ 2 -1  4 -1 -1]

- `feature` : [ 3 -2  2 -2 -2]

- `threshold` : [ 0.75000001 -2.          4.8499999  -2.      -2.        ]

- `impurity` : [0.6616     0.         0.48852158 0.         0.12444444]

- `max_depth` : 2

- `max_n_classes` : 3

- `n_classes` : [3]

- `n_features` : 4

- `n_leaves` : 3

- `n_node_samples` : [50 17 33 18 15]

- `n_outputs` : 1

- `node_count` : 5

- `value` [ [[17. 19. 14.]], 
            [[17.  0.  0.]], 
            [[ 0. 19. 14.]],
            [[ 0. 18.  0.]],
            [[ 0.  1. 14.]]
            ]
            
- `weighted_n_node_samples` [50. 17. 33. 18. 15.]