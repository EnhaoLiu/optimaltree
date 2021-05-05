# Optimal Classification Tree

This is a **python** implementation for two new methods of Optimal Classification Trees proposed by Enhao Liu and Theodore Allen: (1) Modified-Optimal Classification Tree (M-OCT) and (2) BinaryNodePenalty-Optimal Classification Tree (BNP-OCT)


The implementation of `lsopt` is compatible with `scikit-learn` machine learning mudules. See **User Guide** for more details. 

Contact: Enhao Liu (liu.5045@osu.edu), Theodre Allen (allen.515@osu.edu)

## User Guide

The implementation of `lsopt` is compatible with `scikit-learn` machine learning mudules. 

### Example of Iris

Please check the `example_notebook.ipynb` for an example of Iris data set.

```python
import pandas as pd
import numpy as np

from lsopt.tree import OptimalTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


data_path = './data/Iris.csv'

iris_data = pd.read_csv(data_path)
iris_data.drop(columns=["Id"], inplace=True)

X = iris_data.iloc[:, 0:4].to_numpy()
y = iris_data["Species"].to_numpy()


# Define OptimalTreeClassifier
# OCT parameters
max_depth = 3
min_samples_leaf = 1
alpha = 0.01
time_limit = 5  # minute
mip_gap_tol = 0.5  # optimal gap percentage
mip_focus = 'balance'
mip_polish_time = None
warm_start = False
log_file = None

# Construct OCT classifier
oct_model = OptimalTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  alpha=alpha,
                                  criterion="gini",
                                  solver="gurobi",
                                  time_limit=time_limit,
                                  verbose=True,
                                  warm_start=warm_start,
                                  log_file=log_file,
                                  solver_options={'mip_cuts': 'auto',
                                                  'mip_gap_tol': mip_gap_tol,
                                                  'mip_focus': mip_focus,
                                                  'mip_polish_time': mip_polish_time
                                                  }
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
Read LP format model from file /var/folders/5y/drctn5r14mzfjfjnld2ybpc80000gn/T/tmpz_mdbe0o.pyomo.lp
Reading time = 0.03 seconds
x1315: 5081 rows, 1312 columns, 28848 nonzeros
Changed value of parameter TimeLimit to 300.0
   Prev: inf  Min: 0.0  Max: inf  Default: inf
Parameter Cuts unchanged
   Value: -1  Min: -1  Max: 3  Default: -1
Changed value of parameter MIPGap to 0.5
   Prev: 0.0001  Min: 0.0  Max: inf  Default: 0.0001
Parameter MIPFocus unchanged
   Value: 0  Min: 0  Max: 3  Default: 0
Gurobi Optimizer version 9.0.1 build v9.0.1rc0 (mac64)
Optimize a model with 5081 rows, 1312 columns and 28848 nonzeros
Model fingerprint: 0x63cb3227
Variable types: 48 continuous, 1264 integer (1264 binary)
Coefficient statistics:
  Matrix range     [1e-06, 2e+02]
  Objective range  [1e-02, 1e-02]
  Bounds range     [1e+00, 1e+00]
  RHS range        [1e+00, 2e+02]
Presolve removed 316 rows and 5 columns
Presolve time: 0.03s
Presolved: 4765 rows, 1307 columns, 26152 nonzeros
Variable types: 7 continuous, 1300 integer (1260 binary)
Found heuristic solution: objective 0.6700000

Root relaxation: objective 1.000000e-02, 428 iterations, 0.01 seconds

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0    0.01000    0   13    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0  104    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0  125    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   33    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   17    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   60    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   84    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   23    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   53    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   64    0.67000    0.01000  98.5%     -    0s
     0     0    0.01000    0   64    0.67000    0.01000  98.5%     -    0s
     0     2    0.01000    0   64    0.67000    0.01000  98.5%     -    1s
H   83    80                       0.6500000    0.01000  98.5%   128    1s
H  130   126                       0.6400000    0.01000  98.4%  98.2    1s
H  133   126                       0.5800000    0.01000  98.3%   101    1s
H  135   126                       0.5300000    0.01000  98.1%   100    1s
H 1555   798                       0.5100000    0.01000  98.0%  49.4    3s
  1559   800    0.03000   17   75    0.51000    0.01000  98.0%  49.3    5s
H 1561   761                       0.4900000    0.01156  97.6%  49.2    6s
H 1564   725                       0.3700000    0.02000  94.6%  49.2    8s
H 1564   689                       0.2300000    0.02000  91.3%  49.2    8s
H 1565   654                       0.1000000    0.02000  80.0%  49.1    9s
H 1566   622                       0.0800000    0.02000  75.0%  49.1   10s
  1572   626    0.02017   35  395    0.08000    0.02000  75.0%  48.9   15s
H 1578   598                       0.0700000    0.02000  71.4%  48.7   20s
  1583   601    0.03000   26  487    0.07000    0.02000  71.4%  48.6   25s
  1587   604    0.02017   34  690    0.07000    0.02000  71.4%  48.4   30s
  1594   609    0.03000   30  456    0.07000    0.02000  71.4%  48.2   35s
  1600   613    0.06000   63  544    0.07000    0.02000  71.4%  48.0   40s
  1607   617    0.03000   25  688    0.07000    0.02000  71.4%  47.8   45s
  1613   621    0.03000   44  620    0.07000    0.02000  71.4%  47.7   50s
  1619   625    0.02017   20  687    0.07000    0.02000  71.4%  47.5   55s
  1630   634    0.05000   88   34    0.07000    0.02000  71.4%   115   60s
  1636   638    0.03000   27  452    0.07000    0.02000  71.4%   114   65s
  1643   643    0.03000   66  200    0.07000    0.02000  71.4%   114   70s
  1648   649    0.02000   30   50    0.07000    0.02000  71.4%   143   75s
  1654   659    0.02000   32  508    0.07000    0.02000  71.4%   145   81s
  1785   725    0.03000   39  130    0.07000    0.02053  70.7%   164   85s
H 1871   699                       0.0600000    0.02053  65.8%   162   88s
  1949   688    0.03000   49  118    0.06000    0.02053  65.8%   163   90s
  3319   651    0.04012   49  150    0.06000    0.03000  50.0%   144   95s
  5543   841    0.04000   41  214    0.06000    0.03000  50.0%   130  100s
  6565  1022 infeasible   43         0.06000    0.03000  50.0%   137  105s
  7582  1187    0.04000   40  203    0.06000    0.03000  50.0%   146  110s
  8554  1320 infeasible   45         0.06000    0.03000  50.0%   151  116s
  9379  1487    0.04216   42  319    0.06000    0.03000  50.0%   154  121s
 10315  1661    0.04000   51  104    0.06000    0.03000  50.0%   158  125s
 11398  1868     cutoff   40         0.06000    0.03000  50.0%   163  130s
 12162  1979    0.04000   42  229    0.06000    0.03000  50.0%   168  136s
 12765  2092    0.04000   50  138    0.06000    0.03000  50.0%   174  140s
 14101  2249    0.03034   51  180    0.06000    0.03000  50.0%   178  146s
 15029  2299    0.03000   38  296    0.06000    0.03000  50.0%   182  155s
 15429  2399    0.04134   46  211    0.06000    0.03000  50.0%   185  161s
 15887  2472    0.04000   43  252    0.06000    0.03000  50.0%   185  165s

Cutting planes:
  Learned: 31
  Gomory: 6
  Cover: 1421
  Implied bound: 3
  Projected implied bound: 3
  Clique: 21
  MIR: 168
  StrongCG: 23
  Flow cover: 651
  GUB cover: 205
  Inf proof: 47
  Zero half: 23
  RLT: 19
  Relax-and-lift: 25

Explored 16514 nodes (3060169 simplex iterations) in 165.47 seconds
Thread count was 16 (of 16 available processors)

Solution count 10: 0.06 0.07 0.08 ... 0.58

Optimal solution found (tolerance 5.00e-01)
Best objective 6.000000000000e-02, best bound 3.000000000000e-02, gap 50.0000%
Solver running time: 165.6857500076294
Solver termination condition: optimal
Valid Tree : Yes
<oct.tree.OptimalTreeClassifier at 0x7fb1e79df198>


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

