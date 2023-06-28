"""This script is used to run data experiments
that are in the folder "BinNodePenalty-Experiments"
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

from ._base import _get_baseloss, render_plot_tree
import pyomo.environ as pyo
from .tree import OptimalTreeClassifier
from .tree import OldOptimalTreeClassifier
from .tree import BinNodePenaltyOptimalTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
import graphviz


DATA_NAME_LIST = [
    'Balance-scale',
    'Bank market. 10%',
    'Banknote-auth.',
    'Car-evaluation',
    'Ionosphere',
    'Iris',
    'Monks-probl-1',
    'Monks-probl-2',
    'Monks-probl-3',
    'PI-diabetes',
    'Qsar-biodeg.',
    'Seismic-bumps',
    'Spambase',
    'Statlog-sat.',
    'Tic-tac-toe',
    'Wine']

FILE_NAME_LIST = [
    'balance-scale',
    'bank_conv',
    'banknote',
    'car_evaluation',
    'Ionosphere',
    'iris',
    'monk1',
    'monk2',
    'monk3',
    'IndiansDiabetes',
    'biodeg',
    'seismic_bumps',
    'spambase',
    'Statlog_satellite',
    'tic-tac-toe',
    'wine'
]


# %%
def generate_train_val_df(file_name, file_path='./BinNodePenalty-Experiments'):
    data_path = file_path + '/' + file_name + '.csv'
    df = pd.read_csv(data_path, sep=';')
    df.dropna(inplace=True)

    # remove features with all the same values
    cols = df.columns[:-1]
    diff = df[cols].diff().abs().sum()
    df.drop(diff[diff == 0].index, axis=1, inplace=True)

    N = df.shape[0]
    F = df.shape[1] - 1
    C = len(np.unique(df[df.columns[-1]]))

    df_sample = df.sample(frac=0.75, random_state=0)
    df_sample.reset_index(drop=True, inplace=True)
    kf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    kf_split = kf.split(df_sample)

    set_count = 0
    print('Generate Train & Test for Data File = {}'.format(file_name))
    for train_idx, test_idx in kf_split:
        df_train = df_sample.loc[train_idx]
        df_test = df_sample.loc[test_idx]

        train_path = file_path + '/' + file_name + \
            '_train_{}'.format(set_count) + '.csv'
        test_path = file_path + '/' + file_name + \
            '_test_{}'.format(set_count) + '.csv'


#         print('Save Training Set into {}'.format(train_path))
        df_train.to_csv(train_path, sep=';', index=False)

#         print('Save Testing Set into {}'.format(test_path))
        df_test.to_csv(test_path, sep=';', index=False)

        set_count += 1
    return N, F, C


# %%
def load_train_val_df(file_name, file_path='./BinNodePenalty-Experiments', set_idx=0, verbose=True):

    train_path = file_path + '/' + file_name + \
        '_train_{}'.format(set_idx) + '.csv'
    test_path = file_path + '/' + file_name + \
        '_test_{}'.format(set_idx) + '.csv'

    df_train = pd.read_csv(train_path, sep=';')
    df_test = pd.read_csv(test_path, sep=';')

    if verbose:
        print("Load Train & Test dataset {} for data file = {}".format(
            set_idx, file_name))

    return df_train, df_test

# %%


def get_X_y(df):
    X = df.iloc[:, :-1].to_numpy()
    X = X.astype(float)
    y = df[df.columns[-1]].to_numpy()
    return X, y

# %%


def evaluate_tree_model(model, X_train, y_train, X_test, y_test, set_idx, file_name, alpha):
    model_method = type(model).__name__

    if model_method == 'DecisionTreeClassifier':
        run_time = 0
        solution_condition = 'greedy'
        time_limit = 0
    else:
        run_time = model.run_time
        solution_condition = model.solution_condition
        time_limit = 60*model.solver_options['time_limit']

    n_branch_nodes = model.tree_.node_count - model.tree_.n_leaves
    L_hat = _get_baseloss(y_train)
    C = model.n_classes_
    max_depth = model.max_depth
    min_samples_leaf = model.min_samples_leaf

    if C > 2:
        avg = 'micro'

    if C == 2:
        avg = 'binary'

    y_pred_train = model.predict(X=X_train)
    mis_counts_train = sum(y_pred_train != y_train)
    # Re-calculate objective value
    obj_value = mis_counts_train/L_hat + alpha * n_branch_nodes
    # Solver objective value
    if model_method == 'DecisionTreeClassifier':
        obj_value_solver = obj_value
    else:
        obj_value_solver = pyo.value(model.tree_.model.Obj)

    eval_accuracy_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    eval_f1_train = f1_score(y_true=y_train, y_pred=y_pred_train, average=avg)

    y_pred_test = model.predict(X=X_test)
    mis_counts_test = sum(y_pred_test != y_test)
    eval_accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    eval_f1_test = f1_score(y_true=y_test, y_pred=y_pred_test, average=avg)

    # shape outputs
    output_dict = {
        'file_name': [file_name],
        'train_test_set_idx': [set_idx],
        'y_train_true': [y_train],
        'y_train_pred': [y_pred_train],
        'y_test_true': [y_test],
        'y_test_pred': [y_pred_test],
        'model_method': [model_method],
        'run_time': [run_time],
        'time_limit': [time_limit],
        'solution_condition': [solution_condition],
        'max_depth': [max_depth],
        'min_samples_leaf': [min_samples_leaf],
        'n_branch_nodes': [n_branch_nodes],
        'obj_value': [obj_value],
        'obj_value_solver': [obj_value_solver],
        'mis_counts_train': [mis_counts_train],
        'mis_counts_test': [mis_counts_test],
        'accuracy_train': [eval_accuracy_train],
        'accuracy_test': [eval_accuracy_test],
        'f1_train': [eval_f1_train],
        'f1_test': [eval_f1_test]
    }

    output_df = pd.DataFrame(output_dict)

    return output_df

# %%


def run_experiments_CART(file_name,
                         file_path,
                         log_path,
                         max_depth,
                         min_samples_leaf,
                         alpha,
                         train_test_set=[0, 1, 2, 3, 4]
                         ):

    eval_df_all = pd.DataFrame()
    # Loop through train & test set: 0, 1, 2, 3, 4
    for train_test_set_idx in tqdm(train_test_set):
        # load train & testing data
        df_train, df_test = load_train_val_df(file_name=file_name,
                                              file_path=file_path,
                                              set_idx=train_test_set_idx,
                                              verbose=False
                                              )
        X_train, y_train = get_X_y(df_train)
        X_test, y_test = get_X_y(df_test)

        feature_names = df_train.columns.values[:-1]
        class_names = np.unique(y_train)
        class_names = list(class_names.astype(int).astype(str))

        # CART decision tree
        model = tree.DecisionTreeClassifier(max_depth=max_depth,
                                            criterion="gini",
                                            min_samples_leaf=min_samples_leaf,
                                            ccp_alpha=alpha
                                            )

        # Fit on training data
        model.fit(X_train, y_train)

        # # Plot tree: Render & Save the constructed tree
        # plot_file_name = file_name + \
        #     '_train_{}'.format(train_test_set_idx) + '_cart_plot'
        # render_plot_tree(model=model,
        #                  feature_names=feature_names,
        #                  class_names=class_names,
        #                  file_name=plot_file_name,
        #                  file_path=log_path
        #                  )

        # Evaluation dataframe
        eval_df = evaluate_tree_model(model=model,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      set_idx=train_test_set_idx,
                                      file_name=file_name,
                                      alpha=alpha
                                      )
        eval_df_all = pd.concat([eval_df_all, eval_df], ignore_index=True)

    # Save the evaluation dataframe
    df_file_name = file_name + \
        "_eval_cart_df_maxdepth_{}.csv".format(max_depth)
    df_file_path = log_path + '/' + df_file_name
    eval_df_all.to_csv(df_file_path, index=False)

    return eval_df_all


# %%
def run_experiments_OCT(file_name, file_path, log_path,
                        max_depth,
                        min_samples_leaf,
                        alpha,
                        time_limit,
                        mip_gap_tol,
                        mip_polish_time,
                        solver='gurobi',
                        mip_focus='balance',
                        verbose=False,
                        warm_start=False,
                        train_test_set=[0, 1, 2, 3, 4]
                        ):

    eval_df_all = pd.DataFrame()
    # Loop through train & test set: 0, 1, 2, 3, 4

    for train_test_set_idx in tqdm(train_test_set):
        # load train & testing data
        df_train, df_test = load_train_val_df(file_name=file_name,
                                              file_path=file_path,
                                              set_idx=train_test_set_idx,
                                              verbose=False
                                              )
        X_train, y_train = get_X_y(df_train)
        X_test, y_test = get_X_y(df_test)

        feature_names = df_train.columns.values[:-1]
        class_names = np.unique(y_train)
        class_names = list(class_names.astype(int).astype(str))

        # OCT parameters
        if warm_start:
            log_file = log_path + '/' + file_name + \
                '_train_{}'.format(train_test_set_idx) + \
                '_oct_warmstart_logs_maxdepth_{}.txt'.format(max_depth)
        else:
            log_file = log_path + '/' + file_name + \
                '_train_{}'.format(train_test_set_idx) + \
                '_oct_logs_maxdepth_{}.txt'.format(max_depth)

        # Construct OCT classifier
        model = OptimalTreeClassifier(max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      alpha=alpha,
                                      criterion="gini",
                                      solver=solver,
                                      time_limit=time_limit,
                                      verbose=verbose,
                                      warm_start=warm_start,
                                      log_file=log_file,
                                      solver_options={'mip_cuts': 'auto',
                                                      'mip_gap_tol': mip_gap_tol,
                                                      'mip_focus': mip_focus,
                                                      'mip_polish_time': mip_polish_time
                                                      }
                                      )

        # Fit on training data
        model.fit(X_train, y_train)

        # # Plot tree: Render & Save the constructed tree
        # plot_file_name = file_name + \
        #     '_train_{}'.format(train_test_set_idx) + '_oct_plot'
        # render_plot_tree(model=model,
        #                  feature_names=feature_names,
        #                  class_names=class_names,
        #                  file_name=plot_file_name,
        #                  file_path=log_path
        #                  )

        # Evaluation dataframe
        eval_df = evaluate_tree_model(model=model,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      set_idx=train_test_set_idx,
                                      file_name=file_name,
                                      alpha=alpha
                                      )
        eval_df_all = pd.concat([eval_df_all, eval_df], ignore_index=True)

    if warm_start:
        eval_df_all['model_method'] = eval_df_all['model_method'] + \
            ' with Warm Start'

        df_file_name = file_name + \
            "_eval_oct_warmstart_df_maxdepth_{}.csv".format(max_depth)
    else:
        df_file_name = file_name + \
            "_eval_oct_df_maxdepth_{}.csv".format(max_depth)

    # Save the evaluation dataframe
    df_file_path = log_path + '/' + df_file_name
    eval_df_all.to_csv(df_file_path, index=False)

    return eval_df_all

# %%


def run_experiments_OldOCT(file_name, file_path, log_path,
                           max_depth,
                           min_samples_leaf,
                           alpha,
                           epsilon_option,
                           time_limit,
                           mip_gap_tol,
                           mip_polish_time,
                           solver='gurobi',
                           mip_focus='balance',
                           verbose=False,
                           warm_start=False,
                           train_test_set=[0, 1, 2, 3, 4]
                           ):

    eval_df_all = pd.DataFrame()
    # Loop through train & test set: 0, 1, 2, 3, 4
    filename_default = f"old_oct_epsilon_option_{epsilon_option}"
    methodname_default = f"Epsilon Strategy {epsilon_option}"

    for train_test_set_idx in tqdm(train_test_set):
        # load train & testing data
        df_train, df_test = load_train_val_df(file_name=file_name,
                                              file_path=file_path,
                                              set_idx=train_test_set_idx,
                                              verbose=False
                                              )
        X_train, y_train = get_X_y(df_train)
        X_test, y_test = get_X_y(df_test)

        feature_names = df_train.columns.values[:-1]
        class_names = np.unique(y_train)
        class_names = list(class_names.astype(int).astype(str))

        # OCT parameters
        if warm_start:
            log_file = log_path + '/' + file_name + \
                '_train_{}'.format(train_test_set_idx) + \
                f'_{filename_default}_warmstart_logs_maxdepth_{max_depth}.txt'
        else:
            log_file = log_path + '/' + file_name + \
                '_train_{}'.format(train_test_set_idx) + \
                f'_{filename_default}_logs_maxdepth_{max_depth}.txt'

        # Construct OCT classifier
        model = OldOptimalTreeClassifier(max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf,
                                         alpha=alpha,
                                         epsilon_option=epsilon_option,
                                         criterion="gini",
                                         solver=solver,
                                         time_limit=time_limit,
                                         verbose=verbose,
                                         warm_start=warm_start,
                                         log_file=log_file,
                                         solver_options={'mip_cuts': 'auto',
                                                         'mip_gap_tol': mip_gap_tol,
                                                         'mip_focus': mip_focus,
                                                         'mip_polish_time': mip_polish_time
                                                         }
                                         )

        # Fit on training data
        model.fit(X_train, y_train)

        # # Plot tree: Render & Save the constructed tree
        # plot_file_name = file_name + \
        #     '_train_{}'.format(train_test_set_idx) + '_oct_plot'
        # render_plot_tree(model=model,
        #                  feature_names=feature_names,
        #                  class_names=class_names,
        #                  file_name=plot_file_name,
        #                  file_path=log_path
        #                  )

        # Evaluation dataframe
        eval_df = evaluate_tree_model(model=model,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      set_idx=train_test_set_idx,
                                      file_name=file_name,
                                      alpha=alpha
                                      )
        eval_df_all = pd.concat([eval_df_all, eval_df], ignore_index=True)

    if warm_start:
        eval_df_all['model_method'] = eval_df_all['model_method'] + \
            ' with Warm Start' + f" ({methodname_default})"

        df_file_name = file_name + \
            f"_eval_{filename_default}_warmstart_df_maxdepth_{max_depth}.csv"
    else:
        eval_df_all['model_method'] = eval_df_all['model_method'] + \
            f" ({methodname_default})"

        df_file_name = file_name + \
            f"_eval_{filename_default}_df_maxdepth_{max_depth}.csv"

    # Save the evaluation dataframe
    df_file_path = log_path + '/' + df_file_name
    eval_df_all.to_csv(df_file_path, index=False)

    return eval_df_all

# %%


def run_experiments_BinNodePenaltyOCT(file_name, file_path, log_path,
                                      max_depth,
                                      min_samples_leaf,
                                      alpha,
                                      time_limit,
                                      mip_gap_tol,
                                      mip_polish_time,
                                      solver='gurobi',
                                      mip_focus='balance',
                                      verbose=False,
                                      warm_start=False,
                                      train_test_set=[0, 1, 2, 3, 4]
                                      ):

    eval_df_all = pd.DataFrame()
    # Loop through train & test set: 0, 1, 2, 3, 4
    for train_test_set_idx in tqdm(train_test_set):
        # load train & testing data
        df_train, df_test = load_train_val_df(file_name=file_name,
                                              file_path=file_path,
                                              set_idx=train_test_set_idx,
                                              verbose=False
                                              )
        X_train, y_train = get_X_y(df_train)
        X_test, y_test = get_X_y(df_test)

        feature_names = df_train.columns.values[:-1]
        class_names = np.unique(y_train)
        class_names = list(class_names.astype(int).astype(str))

        # OCT parameters
        if warm_start:
            log_file = log_path + '/' + file_name + \
                '_train_{}'.format(train_test_set_idx) + \
                '_bnp_oct_warmstart_logs_maxdepth_{}.txt'.format(max_depth)
        else:

            log_file = log_path + '/' + file_name + \
                '_train_{}'.format(train_test_set_idx) + \
                '_bnp_oct_logs_maxdepth_{}.txt'.format(max_depth)

        # Construct OCT classifier
        model = BinNodePenaltyOptimalTreeClassifier(max_depth=max_depth,
                                                    min_samples_leaf=min_samples_leaf,
                                                    alpha=alpha,
                                                    criterion="gini",
                                                    solver=solver,
                                                    time_limit=time_limit,
                                                    verbose=verbose,
                                                    warm_start=warm_start,
                                                    log_file=log_file,
                                                    solver_options={'mip_cuts': 'auto',
                                                                    'mip_gap_tol': mip_gap_tol,
                                                                    'mip_focus': mip_focus,
                                                                    'mip_polish_time': mip_polish_time
                                                                    }
                                                    )

        # Fit on training data
        model.fit(X_train, y_train)

        # # Plot tree: Render & Save the constructed tree
        # plot_file_name = file_name + \
        #     '_train_{}'.format(train_test_set_idx) + '_bnp_oct_plot'
        # render_plot_tree(model=model,
        #                  feature_names=feature_names,
        #                  class_names=class_names,
        #                  file_name=plot_file_name,
        #                  file_path=log_path
        #                  )

        # Evaluation dataframe
        eval_df = evaluate_tree_model(model=model,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      set_idx=train_test_set_idx,
                                      file_name=file_name,
                                      alpha=alpha
                                      )
        eval_df_all = pd.concat([eval_df_all, eval_df], ignore_index=True)

    if warm_start:
        eval_df_all['model_method'] = eval_df_all['model_method'] + \
            ' with Warm Start'
        df_file_name = file_name + \
            "_eval_bnp_oct_warmstart_df_maxdepth_{}.csv".format(max_depth)
    else:
        df_file_name = file_name + \
            "_eval_bnp_oct_df_maxdepth_{}.csv".format(max_depth)

    # Save the evaluation dataframe
    df_file_path = log_path + '/' + df_file_name
    eval_df_all.to_csv(df_file_path, index=False)

    return eval_df_all
