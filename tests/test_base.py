# %%
from lsopt._base import solve_oct_MILP, _check_preprocess_X_y, ModelTree
import pyomo.environ as pyo
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
# transform X and y

# iris_X_transformed, iris_y_transformed, iris_class_names, \
#     iris_L_hat, iris_epsilons, iris_scaler_X, feature_removed_idx = \
#     _check_preprocess_X_y(X=iris_sub_X, y=iris_sub_Y_label)

iris_X_transformed, iris_y_transformed, iris_class_names, \
    iris_L_hat, iris_epsilons, iris_scaler_X, feature_removed_idx = \
    _check_preprocess_X_y(X=iris_all_X, y=iris_all_Y_label)

# %%
# solve oct
model = solve_oct_MILP(X_transformed=iris_X_transformed,
                       y_transformed=iris_y_transformed,
                       L_hat=iris_L_hat,
                       epsilons=iris_epsilons,
                       max_depth=3,
                       alpha=0.01,
                       min_samples_leaf=1,
                       solver='gurobi',
                       verbose=True,
                       time_limit=1,
                       mip_cuts='auto',
                       mip_gap_tol=0.8,
                       mip_focus="balance"
                       )

# %%
opt_tree = ModelTree(model=model,
                     scaler_X=iris_scaler_X,
                     classes=iris_class_names,
                     feature_removed_idx=feature_removed_idx)

opt_tree.shape(X_transformed=iris_X_transformed,
               y_transformed=iris_y_transformed)

Y_pred = opt_tree.predict(X=iris_all_X, kind="class")

print("Confusion Matrix : \n")
print(confusion_matrix(y_true=iris_all_Y_label,
                       y_pred=Y_pred))

print(classification_report(y_true=iris_all_Y_label,
                            y_pred=Y_pred))

# print(opt_tree.__dict__)

# %%
for v_data in model.component_data_objects(pyo.Var, active=True,
                                           descend_into=True):
    print("Found : {}, value = {}".format(v_data, pyo.value(v_data)))
