# %%
from __future__ import division
import pyomo.environ as pyo
import pyomo.common
import numpy as np
import pandas as pd
import math
import itertools
import warnings
import numbers
from inspect import isclass

from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

from ._exceptions import NotFittedError


# import logging
# logger = logging.getLogger('lsopt._base')
# this script includes several utility functions that will be used in OCT

# %%


def _get_parent(node):
    """Get parent node of a node t

    Parameters
    ----------
    node : int
        The node index of a tree
        NOTE: node starts from 1

    Returns
    -------
    parent : int
        The node index of a tree
    """
    parent = max(math.floor(node / 2), 1)
    return parent


# %%
def _get_ancestors(node, kind="all"):
    """Get a set of ancestors nodes of a node t

    Parameters
    ----------
    node : int
        The node index of a tree
        NOTE: node starts from 1

    kind : str, default='all'
        Define what kind of ancestors.
        If kind='left', it will return a set of ancestors of node t whose left branch has been followed on the path from the root node to t.
        If kind='right', it will return the set of right-branch ancestors.
        If kind = 'all', it will return the set of all ancestors of the given node.

    Returns
    -------
    ancestors : list
        A set of ancestors of a given node
    """
    all_kind = ['all', 'left', 'right']

    if kind not in all_kind:
        raise ValueError(
            "The ancestors only supports kind in {}, got {}.".format(all_kind, kind))

    n_total_parents = math.floor(math.log(node, 2))
    ancestors = list()

    for _ in range(n_total_parents):
        if kind == 'all':
            node = _get_parent(node)
            ancestors.append(node)
        elif kind == 'left' and node % 2 == 0:
            node = _get_parent(node)
            ancestors.append(node)
        elif kind == 'right' and node % 2 != 0:
            node = _get_parent(node)
            ancestors.append(node)
        else:
            node = _get_parent(node)

    return ancestors


# %%
def _get_ancestors_df(node):
    """Get a dataframe including a set of ancestors (left and right) nodes
    for the input node

    Parameters
    ----------
    node : int
        The node index of a tree
        NOTE: node starts from 1

    Returns
    -------
    ancestors_df : dataframe
        A dataframe of ancestors (left and right) of a given node
    """
    ancestors_left = _get_ancestors(node, kind="left")
    direction_left = itertools.repeat("left", len(ancestors_left))

    ancestors_right = _get_ancestors(node, kind="right")
    direction_right = itertools.repeat("right", len(ancestors_right))

    # concatenate list and remove the empty list
    ancestors = list(itertools.chain(ancestors_left, ancestors_right))
    directions = list(itertools.chain(direction_left, direction_right))

    ancestors_dict = dict({"ancestors": ancestors, "direction": directions})
    ancestors_df = pd.DataFrame(ancestors_dict)

    return ancestors_df

# %%


def _get_pair_nodes_ancestors(nodes):
    """Get two lists containing leaf nodes and the corresponding ancestors nodes (left and right).

    Parameters
    ----------
    nodes : list
        A list of leaf nodes indexes of a tree

    Returns
    -------
    pair_nodes_left_ancestors : list of the shape [(leaf_node, ancestors_node)]
        The pairs of leaf nodes and their left-branch ancestors nodes

    pair_nodes_right_ancestors : list of the shape [(leaf_node, ancestors_node)]
        The pairs of leaf nodes and their right-branch ancestors nodes
    """

    if not isinstance(nodes, list):
        raise ValueError("The nodes should be a list.")

    all_ancestors_df = pd.DataFrame()
    for node in nodes:
        ancestors_df = _get_ancestors_df(node=node)
        ancestors_df['nodes'] = node
        all_ancestors_df = pd.concat(
            [all_ancestors_df, ancestors_df], ignore_index=True)

    # left-branch ancestors:
    left_ancestors_df = (all_ancestors_df
                         [lambda x: x['direction'] == 'left']
                         .filter(items=['nodes', 'ancestors'])
                         )

    # right-branch ancestors:
    right_ancestors_df = (all_ancestors_df
                          [lambda x: x['direction'] == 'right']
                          .filter(items=['nodes', 'ancestors'])
                          )

    # convert to a list of containing tuples
    pair_nodes_left_ancestors = list(
        left_ancestors_df.itertuples(index=False, name=None))

    pair_nodes_right_ancestors = list(
        right_ancestors_df.itertuples(index=False, name=None))

    return pair_nodes_left_ancestors, pair_nodes_right_ancestors

# %%


def _get_pair_nodes_parents(nodes):
    """Get a list containing branch nodes and the corresponding parent nodes.

    Parameters
    ----------
    nodes : list
        A list of leaf nodes indexes of a tree

    Returns
    -------
    pair_branch_nodes_parents : list of the shape [(branch_node, parent_node)]
        The pairs of branch nodes and their parent nodes
    """

    if not isinstance(nodes, list):
        raise ValueError("The nodes should be a list.")

    nodes_copy = nodes.copy()
    # remove the root node 1
    if 1 in nodes_copy:
        nodes_copy.remove(1)

    pair_branch_nodes_parents = list()
    for node in nodes_copy:
        parent = _get_parent(node=node)
        pair = (node, parent)
        pair_branch_nodes_parents.append(pair)

    return pair_branch_nodes_parents


# %%
def _get_tree_nodes_set(max_depth):
    """Generate a dictionary of branch and leaf nodes
    given the maximum depth of a tree

    Parameters
    ----------
    max_depth : int
        The maximum depth of a tree

    Returns
    -------
    nodes_set : a dict of shape {'branch nodes': [], 'leaf nodes': []}
        A dict contains branch nodes set, leaf nodes set
    """
    total_number_nodes = np.power(2, max_depth+1) - 1
    temp = math.floor(total_number_nodes/2)

    branch_nodes_set = list(range(1, temp+1))
    leaf_nodes_set = list(range(temp+1, total_number_nodes+1))
    # all_nodes_set = list(range(1, total_number_nodes+1))

    nodes_set = {'branch nodes': branch_nodes_set,
                 'leaf nodes': leaf_nodes_set}

    return nodes_set


# %%
def _get_epsilon(X_mat):
    """Calculate the epsilon values for each feature in the X

    Parameters
    ----------
    X_mat : {array-like, sparse matrix} of shape (n_samples, n_features)
        Traning data

    Returns
    -------
    epsilons : ndarray of shape (n_features,)
        An array contains epsilon values
    """
    X_mat_sort = np.sort(X_mat, axis=0)  # sort the value for each feature
    X_mat_sort_sup = X_mat_sort.copy()

    X_mat_sort = X_mat_sort[:-1, :]  # remove the last row
    X_mat_sort_sup = X_mat_sort_sup[1:, :]  # remove the first row

    diff_mat = X_mat_sort_sup - X_mat_sort
    diff_mat[diff_mat == 0] = np.nan  # replace 0's as np.nan

    # get the min difference values for each feature
    epsilons = np.nanmin(diff_mat, axis=0)

    return epsilons


# %%
def _get_baseloss(Y_vec):
    """Calculate the L_hat which is miss-classification counts
    by simply predicting the most popular class for the entire dataset.

    Parameters
    ----------
    Y_vec : ndarray of shape (n_samples,)
        Array of labels.

    Returns
    -------
    L_hat : int
        miss-classification counts by simply predicting the most popular class for the entire dataset.
    """
    n_total = Y_vec.shape[0]
    # calculate the counts of each unique label
    counts = np.unique(Y_vec, return_counts=True)[1]
    n_common = counts.max()
    L_hat = n_total - n_common
    return L_hat

# %%


def _get_penalty_nodes(max_depth):
    """Get the weights penalty of branching nodes

    If a specific feature Xj in the desired sets is not being 
    added in the tree, then add penalty weight in the objective function

    Parameters
    ----------
    max_depth : int
        The maximum depth of a tree

    Returns
    -------
    penalty_weights : ndarray of shape (max_nodes, )
    """


# %%
def _transform_X(X_mat, feature_range=(0, 1)):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    The transformation is calculated as::

        X_scaled = scale * X + min - X.min(axis=0) * scale
        where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

    Parameters
    ----------
    X_mat : {array-like, sparse matrix} of shape (n_samples, n_features)
        Traning data

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    Returns
    -------
    X_mat_transform : {array-like, sparse matrix} of shape (n_samples, n_features)
        Normalized training data
    """
    scaler = MinMaxScaler(feature_range=feature_range, copy=True)

    X_mat_transform = scaler.fit_transform(X_mat)

    return X_mat_transform, scaler


# %%
def _transform_y(Y_vec, neg_label=-1, pos_label=1):
    """Transform labels in one-vs-all fashion with 1 and -1 values.

    Parameters
    ----------
    Y_vec : ndarray of shape (n_samples,)
        Array of labels.

    Returns
    -------
    Y_vec_transform : ndarray of shape [n_samples, n_class]
        Target values. The 2-d matrix should only contain -1 and 1, represents multilabel classification.
    classes : array of shape [n_class]
        Holds the label for each class.
    """

    scaler = LabelBinarizer(neg_label=neg_label,
                            pos_label=pos_label,
                            sparse_output=False)

    Y_vec_transform = scaler.fit_transform(Y_vec)

    classes = scaler.classes_

    if classes.size == 2:
        # NOTE: if binary classes, Y = -1 would be the first class, Y = 1 would be the second class
        # Therefore, when forming the Y matrix, we need to reverse the Y values
        temp = np.zeros(shape=(Y_vec.shape[0], 2), dtype=int)
        temp[:, 0] = -Y_vec_transform.ravel()
        temp[:, 1] = Y_vec_transform.ravel()
        Y_vec_transform = temp

    return Y_vec_transform, classes

# %%


def _transform_array2dict(array):
    """Transform 2D array to be a dictionary of shape { (i, j) : value }

    The (i, j) represents a tuple of indexes in the 2D array.

    Note that the i and j will start from 1 instead of 0. For example,
    {(1, 1) :  0.5} represents that the value in the first row and first column
    of the array is 0.5.

    Parameters
    ----------
    array : 2D array of shape (n_rows, n_columns)
        Input object to convert.

    Returns
    -------
    array_transformed : a dictionary of shape { (i, j) : value }
        The converted and validated array.
    """
    # check if array is 2D and make values to be all numeric
    array = check_array(array, dtype="numeric", ensure_2d=True)

    # convert to be the dictionary with indexes and values
    array_transformed = dict()
    for index, value in np.ndenumerate(array):
        add_one_tuple = (1, 1)
        index_new = tuple(map(lambda x: x[0] + x[1],
                              zip(index, add_one_tuple)))
        array_transformed[index_new] = value

    return array_transformed


# %%
def _check_preprocess_X_y(X, y):
    """Preprocess X and y for the inputs of Optimal Classification Tree model:

    Normalize the values of predicted variables (features) X in the data to be [0,1]

    Ignore some predicted variables (features) if the difference between
    their maximum value and minimum value is close to 0.

    Transform the labels of Y to be (n_samples, n_class) shape with -1 and 1 values

    Calculate epsilon values and baseloss value

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector with numeric values, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X. It could be numeric labels or categorical labels.

    Returns
    -------

    X_transformed : nd-array of shape (n_samples, n_features)
        Transformed training data.

    y_transformed : nd-array of shape (n_samples, n_class)
        Transformed target data.

    class_names : nd-array of shape (n_class,)
        The original classes names.

    L_hat : int
        The base loss, i.e., the misclassified counts by simply predicting common labels.

    epsilons : nd-array of shape (n_features,)
        The epsilons values for each feature

    scaler_X : MinMaxScaler object
        The MinMaxScaler() for transforming the training data. It will be
        used to recover the original value by ".inverse_transform" method.

    feature_removed_idx : 1D array
        The removed features' indexes

    """
    # Check and return X and y with valid array shape
    X = check_array(X, dtype="numeric", ensure_2d=True)
    y = check_array(y, dtype=None, ensure_2d=False)
    check_classification_targets(y)

    # Get baseloss (L_hat)
    L_hat = _get_baseloss(y)

    # Transform Y to be -1 and 1 nd-array (n_samples, n_class)
    y_transformed, class_names = _transform_y(y, neg_label=-1, pos_label=1)

    # Remove some features in X if their min and max are the same.
    data_min = np.nanmin(X, axis=0)
    data_max = np.nanmax(X, axis=0)
    diff = data_max - data_min
    feature_removed_idx = np.where(diff == 0)[0]

    # Remove useless features if exist
    if feature_removed_idx.size > 0:
        X = np.delete(X, feature_removed_idx, axis=1)

        warnings.warn("The features in {} columns of the training data"
                      " have been removed because the difference between"
                      " their maximum and minium values is zero."
                      .format(feature_removed_idx))

    # Transform X to be in the range [0,1] (n_samples, n_features)
    X_transformed, scaler_X = _transform_X(X, feature_range=(0, 1))

    # Caculate epsilons for each feature
    epsilons = _get_epsilon(X_transformed)

    return X_transformed, y_transformed, class_names, L_hat, epsilons, scaler_X, feature_removed_idx

# %%


def _check_parameters(alpha=None, max_depth=None, min_samples_leaf=None,
                      criterion=None, time_limit=None):
    """Check if the hyper-parameters in the Optimal Classification Tree are valid

    If any of these parameters are not invalid, it will present the error message.

    Parameters
    ----------
    alpha : float, default=None
        The complexity parameter. If it becomes larger, it will result in less splits.

    max_depth : int, default=None
        The maximum depth of the tree.

    min_samples_leaf : int, default=None
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

    criterion : str, default=None
        The function to measure the quality of a split. 
        Supported criteria are:
            "gini" for the Gini impurity
            "entropy" for the information gain.

    time_limit : int, default=None
        The time limit in minutes of running solver.

    """
    # check alpha
    if alpha:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("'alpha' must be any number from 0.0 to 1.0, got {}"
                             .format(alpha))

    # check max_depth
    if max_depth:
        if isinstance(max_depth, numbers.Integral):
            if max_depth <= 0:
                raise ValueError(
                    "'max_depth' must be positive integer, got {}".format(max_depth))
        else:
            raise ValueError(
                "'max_depth' must be positive integer, got {}".format(max_depth))

    # check min_smaples_leaf
    if min_samples_leaf:
        if isinstance(min_samples_leaf, numbers.Integral):
            if min_samples_leaf <= 0:
                raise ValueError(
                    "'min_samples_leaf' must be positive integer, got{}".format(min_samples_leaf))
        else:
            raise ValueError(
                "'min_samples_leaf' must be positive integer, got{}".format(min_samples_leaf))

    # check criterion
    if criterion:
        all_criterion = ["gini", "entropy"]
        if isinstance(criterion, str):
            if criterion not in all_criterion:
                raise ValueError(
                    "'criterion' must be either 'gini' or 'entropy'.")
        else:
            raise ValueError("'criterion' must be either 'gini' or 'entropy'.")

    # Check time_limit
    if time_limit:
        if not isinstance(time_limit, numbers.Integral) or time_limit <= 0:
            raise ValueError(
                "'time_limit' must be integer value greater than 0.")


# %%


def _check_configure_solver(solver, return_config=True, **kwargs):
    """Check if the solver is accepted in Optimal Classification tree

    Configure the solver if additional options are provided

    * It supports several solvers in ["glpk", "cbc","cplex", "gurobi"]

    Parameters
    ----------
    solver : str, support solvers in ["glpk", "cplex", "gurobi", "cbc"]
        The optimization solver to use.

    return_config : boolean, default=True
        Whether or not return the configured solver. 
        If False, it only check if solver and additional arguments are valid

    Additional Arguments : use to configure additional options in the solver

        time_limit : int, default=5
            The time limit in minutes of running solver. Default is 5 minutes.
            It is used to early stop the solver.

        mip_cuts : str or list, default=None
            The cutting planes are generated by specifying cuts' names or strategies.
            default=None means that use the default setting

            Note: different solvers could support different types of cuts.

            solver="glpk":
                available mip_cuts are in ["gomory", "mir", "cover", "clique"]
                mip_cuts="all" : adding all available cuts
                mip_cuts=["gomory"] : adding only Gomory cuts
                mip_cuts=["gomory", "mir"] : adding Gomory and MIR cuts

            solver="gurobi":
                There is a wide range of cutting plane strategies: 
                https://www.gurobi.com/documentation/9.0/refman/parameters.html#sec:Parameters
                Here, only provide a Global Cut Control setting based on Gurobi Documentation:
                https://www.gurobi.com/documentation/9.0/refman/cuts.html#parameter:Cuts

                mip_cuts="auto" : the default strategy for cuts, it will automatically generate cuts
                mip_cuts="off" : shut off cuts
                mip_cuts="moderate" : moderate cut generation
                mip_cuts="aggressive" : aggressive cut generation

            solver="cplex": 
                There is a plenty of cutting plane strategies:
                https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/UsrMan/topics/discr_optim/mip/cuts/41_params.html#User_manual.uss_solveMIP.672903__title1310474232097
                Here, only provide a Global Cut Control 

                mip_cuts="auto" : the default strategy for cuts, it will automatically generate cuts
                mip_cuts="off" : shut off cuts
                mip_cuts="moderate" : moderate cut generation
                mip_cuts="aggressive" : aggressive cut generation

            solver="cbc":
                available custs are in [XXXXXXXXXXXX]

        mip_gap_tol: float, default=None
            Relative MIP gap tolerance. Any number from 0.0 to 1.0.
            default=None means that using the solver's default MIP gap tolerance.
            For example, mip_gap_tol=0.05 is to enable solver to stop as soon as
            it has found a feasible integer solution proved to be within
            five percent of optimal.

        mip_focus : str, default=None
            NOTE: This option is only avaiable for both solver="gurobi" and solver="cplex".
            Reference: Gurobi: https://www.gurobi.com/documentation/9.0/refman/mipfocus.html#parameter:MIPFocus
                       Cplex : https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/UsrMan/topics/discr_optim/mip/usage/10_emph_feas.html

            solver="gurobi":

                mip_focus="balance" : By default, the Gurobi MIP solver strikes a balance between finding 
                    new feasible solutions and proving that the current solution is optimal.

                mip_focus="feasible" : More interested in finding feasible solutions quickly.

                mip_focus="optimal" : If you believe the solver is having no trouble finding good quality solutions, 
                    and wish to focus more attention on proving optimality.

                mip_focus="bound" : If the best objective bound is moving very slowly (or not at all), 
                    you may want to try to focus on the bound.

            solver="cplex":

                mip_focus="balance" : By default, the Cplex MIP solver strikes a balance between finding 
                    new feasible solutions and proving that the current solution is optimal.

                mip_focus="feasible" : More interested in finding feasible solutions quickly. 
                    User may want a greater emphasis on feasibility and less emphasis on analysis and proof of optimality.

                mip_focus="optimal" : If you believe the solver is having no trouble finding good quality solutions, 
                    and wish to focus more attention on proving optimality.

                mip_focus="bound" : If the best objective bound is moving very slowly (or not at all), 
                    you may want to try to focus on the bound.

                mip_focus="hidden" : This choice is intended for use on difficult models where a proof of optimality is unlikely, 
                    and where mip_focus="feasible" does not deliver solutions of an appropriately high quality.

        fp_heur : boolean, default=False
            Chooses whether or not to apply the Feasibility Pump heuristic on
            finding a feasible solution.

            if solver="glpk", fp_heur's parameter name is 'fpump'

            if solver="cplex", fp_heur's parameter name is 'fpheur'

            if solver="gurobi", fp_heur's parameter name is [XXXXXXXXXXXX]

        backtrack : str, default=None

            This backtracking setting is only for solver="glpk". Available options are
                * "dfs" : backtrack using depth first search
                * "bfs" : backtrack using breadth first search
                * "bestp" : backtrack using the best projection heuristic
            if None, backtrack="bestb" ("glpk" default setting): 
                backtrack using node with best local bound


    Returns
    -------
    solver_configured : object, pyomo.SolverFactory()
        A pyomo.SolverFactory object with additional solver configurations
    """
    # Check solver

    all_solvers = ['glpk', 'cbc', 'cplex', 'gurobi']

    if solver not in all_solvers:
        raise ValueError("Optimal Classification Tree supports only solvers in {}, got {}"
                         .format(all_solvers, solver))

    # Default value for the additional arguments
    time_limit = kwargs.pop('time_limit', 5)
    mip_cuts = kwargs.pop('mip_cuts', None)
    mip_gap_tol = kwargs.pop('mip_gap_tol', None)

    mip_focus = kwargs.pop('mip_focus', None)
    fp_heur = kwargs.pop('fp_heur', False)
    backtrack = kwargs.pop('backtrack', None)

    # Check time_limit
    if not isinstance(time_limit, numbers.Integral) or time_limit <= 0:
        raise ValueError("'time_limit' must be integer value greater than 0.")

    # time limit in seconds
    time_limit = 60*time_limit

    # Check mip_cuts
    if mip_cuts:
        if solver == "glpk":
            mip_cuts_names = ["gomory", "mir", "cover", "clique"]
            if not isinstance(mip_cuts, (str, list)):
                raise ValueError("For solver '{}', 'mip_cuts' is either 'all' or "
                                 "a list contains cuts names in {}"
                                 .format(solver, mip_cuts_names))

            if isinstance(mip_cuts, str):
                if mip_cuts != "all":
                    raise ValueError("For solver '{}', 'mip_cuts' is either 'all' or "
                                     "a list contains cuts names in {}"
                                     .format(solver, mip_cuts_names))
            elif isinstance(mip_cuts, list):
                for cut in mip_cuts:
                    if cut not in mip_cuts_names:
                        raise ValueError("{} is not valid cut."
                                         " The available cuts in solver '{}' are {}."
                                         " Also, adding all available cuts"
                                         " by specifying mip_cuts='all'."
                                         .format(cut, solver, mip_cuts_names)
                                         )

        elif solver == "gurobi" or solver == "cplex":
            global_cuts_strategies = ["auto", "off", "moderate", "aggressive"]
            if not isinstance(mip_cuts, str):
                raise ValueError("For solver '{}', 'mip_cuts' is a string in {}"
                                 .format(solver, global_cuts_strategies))

            if mip_cuts not in global_cuts_strategies:
                raise ValueError("For solver '{}', 'mip_cuts' is a string in {}"
                                 ", got{}"
                                 .format(solver, global_cuts_strategies, mip_cuts))

        # elif solver == "cbc":
        #     # need further works to figure out

    # Check mip_gap_tol
    if mip_gap_tol:
        if mip_gap_tol < 0 or mip_gap_tol > 1:
            raise ValueError("'mip_gap_tol' must be any number from 0.0 to 1.0, "
                             "got {}.".format(mip_gap_tol))

    # Check mip_focus
    if mip_focus:
        if solver == "gurobi":
            all_focus = ["balance", "feasible", "optimal", "bound"]
            if mip_focus not in all_focus:
                raise ValueError("For solver '{}', 'mip_focus' is a string in {}"
                                 ", got {}."
                                 .format(solver, all_focus, mip_focus))
        elif solver == "cplex":
            all_focus = ["balance", "feasible", "optimal", "bound", "hidden"]
            if mip_focus not in all_focus:
                raise ValueError("For solver '{}', 'mip_focus' is a string in {}"
                                 ", got {}."
                                 .format(solver, all_focus, mip_focus))

    # Check fp_heur
    if not isinstance(fp_heur, bool):
        raise ValueError("'fp_heur' must be True or False, "
                         "got {}.".format(fp_heur))

    # Check backtrack
    if backtrack:
        all_backtrack = ['dfs', 'bfs', 'bestp', 'bestb']
        if backtrack not in all_backtrack:
            raise ValueError(
                "backtrack is either None or in {}".format(all_backtrack))

    # ========= Configure the solver ==============
    if return_config:
        solver_configured = pyo.SolverFactory(solver)

        # ==== Configure other options for the solver="glpk" =======

        # List all options for "glpk":
        # In Command Line:
        #   glpsol --help
        if solver == 'glpk':
            # Set time limit
            solver_configured.options['tmlim'] = time_limit

            # Set cuts
            if mip_cuts:
                if isinstance(mip_cuts, str):
                    # Add the all cuts
                    solver_configured.options['cuts'] = None

                elif isinstance(mip_cuts, list):
                    for cut in mip_cuts:
                        solver_configured.options[cut] = None

            # Set relative MIP gap tolerance
            if mip_gap_tol:
                solver_configured.options['mipgap'] = mip_gap_tol

            # Set if apply the Feasibility Pump heuristic
            if fp_heur:
                solver_configured.options['fpump'] = None

            # Set backtracking:
            if backtrack:
                solver_configured.options[backtrack] = None

        # ==== Configure options for the solver="gurobi" =======
        # List all options for "gurobi":
        # In Command Line:
        #   exec gurobi.sh
        #   help(GRB.param)
        # References: https://www.gurobi.com/documentation/9.0/refman/parameters.html#sec:Parameters
        elif solver == 'gurobi':
            # Set time limit
            solver_configured.options['TimeLimit'] = time_limit

            # Set cuts
            if mip_cuts:
                if mip_cuts == 'auto':
                    solver_configured.options['Cuts'] = -1
                elif mip_cuts == 'off':
                    solver_configured.options['Cuts'] = 0
                elif mip_cuts == 'moderate':
                    solver_configured.options['Cuts'] = 1
                elif mip_cuts == 'aggressive':
                    solver_configured.options['Cuts'] = 2

            # Set relative MIP gap tolerance
            if mip_gap_tol:
                solver_configured.options['MIPGap'] = mip_gap_tol

            # Set high-level solution strategy
            if mip_focus:
                if mip_focus == "balance":
                    solver_configured.options['MIPFocus'] = 0
                elif mip_focus == "feasible":
                    solver_configured.options['MIPFocus'] = 1
                elif mip_focus == "optimal":
                    solver_configured.options['MIPFocus'] = 2
                elif mip_focus == "bound":
                    solver_configured.options['MIPFocus'] = 3

        # ==== Configure options for the solver="cplex" =======
        elif solver == 'cplex':
            # Set time limit
            solver_configured.options['timelimit'] = time_limit

            # Set cuts
            if mip_cuts:
                if mip_cuts == 'auto':
                    solver_configured.options['mip cuts all'] = 0
                elif mip_cuts == 'off':
                    solver_configured.options['mip cuts all'] = -1
                elif mip_cuts == 'moderate':
                    solver_configured.options['mip cuts all'] = 1
                elif mip_cuts == 'aggressive':
                    solver_configured.options['mip cuts all'] = 2

            # Set relative MIP gap tolerance
            if mip_gap_tol:
                solver_configured.options['mip tolerances mipgap'] = mip_gap_tol

            # Set high-level solution strategy
            if mip_focus:
                if mip_focus == "balance":
                    solver_configured.options['emphasis mip'] = 0
                elif mip_focus == "feasible":
                    solver_configured.options['emphasis mip'] = 1
                elif mip_focus == "optimal":
                    solver_configured.options['emphasis mip'] = 2
                elif mip_focus == "bound":
                    solver_configured.options['emphasis mip'] = 3
                elif mip_focus == "hidden":
                    solver_configured.options['emphasis mip'] = 4

        # ==== Configure options for the solver="cbc" =======
        elif solver == 'cbc':
            solver_configured.options['sec'] = time_limit

            # need to add further configurations

        return solver_configured

# %%


def _check_solver_termination(solver_results):
    """Check the solver's termination condition
    and raise errors if there is no feasible solution or optimal solution obtained.

    Parameters
    ----------
    solver_results :
        The results obtained by calling solver.solve(model)
    """

    solver_status = str(solver_results.solver.status)
    solver_termination_condition = str(
        solver_results.solver.termination_condition)

    if solver_status == "ok" or solver_status == "aborted":
        accept_condition = ["feasible", "optimal",
                            'maxTimeLimit', 'maxIterations']

        if solver_termination_condition in accept_condition:
            print("Solver running time: {}".format(
                solver_results.solver.time))

            print("Solver termination condition: {}".format(
                solver_termination_condition))

        else:
            raise RuntimeError("No feasible solution found. The solver termination condition : {}"
                               .format(solver_termination_condition))

    else:
        raise RuntimeError("Solver status : {}. Solver termination condition : {}"
                           " Check here: http://www.pyomo.org/blog/2015/1/8/accessing-solver"
                           .format(solver_status, solver_termination_condition))

# %%


def _check_solution(model_solved):
    """Check if the solution obtained are a valid tree.

    If the complexity parameter alpha is too large, it will
    result in that there is no splits in the decision tree.

    Parameters
    ----------
    model_solved : a pyomo.ConcreteModel object
        a solved model which contains solutions

    """
    if not isinstance(model_solved, pyo.ConcreteModel):
        raise ValueError("'model_solved' must be a pyomo.ConcreteModel")

    # Check if the root node 1 contains split.
    if pyo.value(model_solved.d[1]) == 0:
        raise RuntimeError("Tree is not constructed because of no splits obtained."
                           " Suggestion: Reduce the magnitude of 'alpha' complexity parameter.")
    else:
        print("Valid Tree : Yes")


# %%


def solve_oct_MILP(X_transformed, y_transformed, L_hat, epsilons,
                   alpha=0.01, max_depth=2, min_samples_leaf=1,
                   solver="gurobi", verbose=False, **kwargs):
    """Solve the Optimal Classification Tree by Mixed-Integer Linear Programming

    Parameters
    ----------
    X_transformed : nd-array of shape (n_samples, n_features)
        Transformed training data.

    y_transformed : nd-array of shape (n_samples, n_class)
        Transformed target data.

    L_hat : int
        The base loss, i.e., the misclassified counts by simply predicting common labels.

    epsilons : nd-array of shape (n_features,)
        The epsilons values for each feature

    alpha : float, default=0.01
        The complexity parameter. Any number from 0.0 to 1.0.
        If it becomes larger, it will result in less splits.
        if alpha=0.0, the tree will be a full tree and might result in overfitting model.

    max_depth : int, default=2
        The maximum depth of the tree.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

    solver : str, default="gurobi"
        The optimization solver to use.

        It supports several solvers in ["cplex", "gurobi", "glpk", "cbc"]

    verbose : boolean, default=False
        If True, display the outputs of the solver

    **kwargs : additional arguments that use to configure additional options in the solver

        time_limit : int, default=5
            The time limit in minutes of running solver. Default is 5 minutes.
            It is used to early stop the solver.

        mip_cuts : str or list, default=None
            The cutting planes are generated by specifying cuts' names or strategies.
            default=None means that use the default setting

            NOTE: different solvers could support different types of cuts.

            solver="glpk":
                available mip_cuts are in ["gomory", "mir", "cover", "clique"]
                mip_cuts="all" : adding all available cuts
                mip_cuts=["gomory"] : adding only Gomory cuts
                mip_cuts=["gomory", "mir"] : adding Gomory and MIR cuts

            solver="gurobi":
                There is a wide range of cutting plane strategies: 
                https://www.gurobi.com/documentation/9.0/refman/parameters.html#sec:Parameters
                Here, we only provide a Global Cut Control setting based on Gurobi Documentation:
                https://www.gurobi.com/documentation/9.0/refman/cuts.html#parameter:Cuts

                mip_cuts="auto" : the default strategy for cuts, it will automatically generate cuts
                mip_cuts="off" : shut off cuts
                mip_cuts="moderate" : moderate cut generation
                mip_cuts="aggressive" : aggressive cut generation

            solver="cplex": 
                There is a plenty of cutting plane strategies:
                https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/UsrMan/topics/discr_optim/mip/cuts/41_params.html#User_manual.uss_solveMIP.672903__title1310474232097
                Here, only provide a Global Cut Control 

                mip_cuts="auto" : the default strategy for cuts, it will automatically generate cuts
                mip_cuts="off" : shut off cuts
                mip_cuts="moderate" : moderate cut generation
                mip_cuts="aggressive" : aggressive cut generation

            solver="cbc":
                available custs are in [XXXXXXXXXXXX]


        mip_gap_tol: float, default=None
            Relative MIP gap tolerance. Any number from 0.0 to 1.0.
            default=None means that using the solver's default MIP gap tolerance.
            For example, mip_gap_tol=0.05 is to enable solver to stop as soon as
            it has found a feasible integer solution proved to be within
            five percent of optimal.

        mip_focus : str, default=None
            NOTE: This option is only avaiable for both solver="gurobi" and solver="cplex".
            Reference: Gurobi: https://www.gurobi.com/documentation/9.0/refman/mipfocus.html#parameter:MIPFocus
                       Cplex : https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/UsrMan/topics/discr_optim/mip/usage/10_emph_feas.html

            solver="gurobi":

                mip_focus="balance" : By default, the Gurobi MIP solver strikes a balance between finding 
                    new feasible solutions and proving that the current solution is optimal.

                mip_focus="feasible" : More interested in finding feasible solutions quickly.

                mip_focus="optimal" : If you believe the solver is having no trouble finding good quality solutions, 
                    and wish to focus more attention on proving optimality.

                mip_focus="bound" : If the best objective bound is moving very slowly (or not at all), 
                    you may want to try to focus on the bound.

            solver="cplex":

                mip_focus="balance" : By default, the Cplex MIP solver strikes a balance between finding 
                    new feasible solutions and proving that the current solution is optimal.

                mip_focus="feasible" : More interested in finding feasible solutions quickly. 
                    User may want a greater emphasis on feasibility and less emphasis on analysis and proof of optimality.

                mip_focus="optimal" : If you believe the solver is having no trouble finding good quality solutions, 
                    and wish to focus more attention on proving optimality.

                mip_focus="bound" : If the best objective bound is moving very slowly (or not at all), 
                    you may want to try to focus on the bound.

                mip_focus="hidden" : This choice is intended for use on difficult models where a proof of optimality is unlikely, 
                    and where mip_focus="feasible" does not deliver solutions of an appropriately high quality.

        fp_heur : boolean, default=False
            Chooses whether or not to apply the Feasibility Pump heuristic on
            finding a feasible solution.

            if solver="glpk", fp_heur's parameter name is 'fpump'

            if solver="cplex", fp_heur's parameter name is 'fpheur'

            if solver="gurobi", fp_heur's parameter name is [XXXXXXXXXXXX]

        backtrack : str, default=None

            This backtracking setting is only for solver="glpk". Available options are
                * "dfs" : backtrack using depth first search
                * "bfs" : backtrack using breadth first search
                * "bestp" : backtrack using the best projection heuristic
            if None, backtrack="bestb" ("glpk" default setting): 
                backtrack using node with best local bound

    Returns
    -------
    model :  a class object of pyomo.ConcreteModel
        A solved model of the Optimal Classification Tree.

        Including solutions' values and constraints' values.

    solver_results : a class object of pyomo.SolverResults
        Including Problem Information, Solver Information,
        and Solution Information.

        solver_results.write() could show the results.

        NOTE: solver_results are used to extract
        the solver termination condition (feasible or optimal),
        and solver's running time.
    """
    # ======================================================================
    # ================CHECK HYPER-PARAMETERS and SOLVER=====================
    # ======================================================================

    # check if hyper-parameters are valid
    _check_parameters(alpha=alpha, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf)

    # check and configure solver
    solver_configured = _check_configure_solver(
        solver=solver, return_config=True, **kwargs)

    # ======================================================================
    # ============================= PREPARE DATA ===========================
    # ======================================================================

    # =========================== Get Node Indexes =========================
    # get the number of samples and number of features
    n_samples, n_features = X_transformed.shape

    # get the number of class
    n_class = y_transformed.shape[1]

    # get the nodes set {'branch nodes':[], 'leaf nodes':[]}: branch nodes and leaf nodes
    nodes_set = _get_tree_nodes_set(max_depth=max_depth)

    # get the pairs of leaf nodes and their ancestors nodes (left and right)
    pair_nodes_left_ancestors, pair_nodes_right_ancestors = _get_pair_nodes_ancestors(
        nodes=nodes_set['leaf nodes'])

    # get the pairs of branch nodes and their parent nodes
    pair_branch_nodes_parents = _get_pair_nodes_parents(
        nodes=nodes_set['branch nodes'])

    # get the pairs of leaf nodes and their parent nodes
    pair_leaf_nodes_parents = _get_pair_nodes_parents(
        nodes=nodes_set['leaf nodes'])

    # ========================== Get Parameters ==============================
    # convert X_transformed and y_transformed to be a dictionary with indexes (row, col) and values
    X_transformed_dict = _transform_array2dict(array=X_transformed)
    y_transformed_dict = _transform_array2dict(array=y_transformed)

    # get the min and max of epsilons
    epsilon_min = np.min(epsilons)
    epsilon_max = np.max(epsilons)

    temp_small_number = 0.00001

    if epsilon_min < temp_small_number:
        epsilon_min = temp_small_number

    if epsilon_max < temp_small_number:
        epsilon_max = 1.5 * temp_small_number

    epsilons_dict = dict(enumerate(epsilons.flatten(), 1))

    # ======================================================================
    # ========================= CONSTRUCT MILP MODEL =======================
    # ======================================================================

    # ========================== Create A Model ============================
    model = pyo.ConcreteModel(name='Optimal Classification Tree')

    # ========================== Define Indexes ============================
    # branch nodes indexes: tB
    model.tB = pyo.Set(initialize=nodes_set['branch nodes'])

    # leaf nodes indexes: tL
    model.tL = pyo.Set(initialize=nodes_set['leaf nodes'])

    # pairs of leaf nodes and their left-branch ancestors nodes: (tL, A_L(tL))
    model.tL_AL = pyo.Set(dimen=2, initialize=pair_nodes_left_ancestors)

    # paris of leaf nodes and their right-branch ancestors nodes : (tL, A_R(tL))
    model.tL_AR = pyo.Set(dimen=2, initialize=pair_nodes_right_ancestors)

    # paris of branch nodes and their parent nodes indexes: (tB, P(tB))
    model.tB_P = pyo.Set(dimen=2, initialize=pair_branch_nodes_parents)

    # paris of leaf nodes and their parent nodes indexes: (tL, P(tL))
    model.tL_P = pyo.Set(dimen=2, initialize=pair_leaf_nodes_parents)

    # samples indexes: I
    model.I = pyo.RangeSet(1, n_samples)

    # features indexes: J
    model.J = pyo.RangeSet(1, n_features)

    # class indexes: K
    model.K = pyo.RangeSet(1, n_class)

    # ========================= Define Parameters ============================
    # data parameters: X[i,j] and y[i,k]
    model.X = pyo.Param(model.I, model.J,
                        initialize=X_transformed_dict)

    model.y = pyo.Param(model.I, model.K,
                        initialize=y_transformed_dict)

    # epsilon parameters: epsilon_min and epsilon_max
    model.epsilon_min = pyo.Param(initialize=epsilon_min)

    model.epsilon_max = pyo.Param(initialize=epsilon_max)

    # FIXME
    model.epsilons = pyo.Param(model.J, initialize=epsilons_dict)

    # baseloss parameter: L_hat
    model.L_hat = pyo.Param(initialize=L_hat)

    # number of samples parameter: n
    model.n = pyo.Param(initialize=n_samples)

    # complexity parameter: alpha
    model.alpha = pyo.Param(initialize=alpha)

    # minimum samples of leaf node parameter: min_samples_leaf
    model.min_samples_leaf = pyo.Param(initialize=min_samples_leaf)

    # ======================== Define Variables ===============================
    # splits coefficient: a[j,tB], binary
    model.a = pyo.Var(model.J, model.tB, domain=pyo.Binary)

    # splits threshold: b[tB], float, bounds [0,1]
    model.b = pyo.Var(model.tB,
                      domain=pyo.NonNegativeReals, bounds=(0, 1))

    # splits indicator: d[tB], binary
    model.d = pyo.Var(model.tB, domain=pyo.Binary)

    # indicate if point i is assigned to leaf node t: z[i, tL], binary
    model.z = pyo.Var(model.I, model.tL, domain=pyo.Binary)

    # indicate if leaf node t contains any point: l[tL], binary
    model.l = pyo.Var(model.tL, domain=pyo.Binary)

    # indicates if class prediction k is assigned to leaf node t: c[k, tL], binary
    model.c = pyo.Var(model.K, model.tL, domain=pyo.Binary)

    # misclassification loss in leaf node: Loss[tL], float, bounds [0, +inf]
    model.Loss = pyo.Var(model.tL, domain=pyo.NonNegativeReals)

    # number of points of label k in leaf node t: NN[k, tL], float, bounds [0, +inf]
    model.NN = pyo.Var(model.K, model.tL,
                       domain=pyo.NonNegativeReals)

    # number of points in leaf node t: N[tL], float, bounds [0, +inf]
    model.N = pyo.Var(model.tL, domain=pyo.NonNegativeReals)

    # ======================== Define Objective ===============================
    def obj_rule(model):
        model_loss = 1/model.L_hat * sum(model.Loss[t] for t in model.tL)
        model_complexity = model.alpha * sum(model.d[t] for t in model.tB)
        return model_loss + model_complexity

    model.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ======================== Define Constraints =============================
    # Loss in leaf node is equal to the number of points in the node
    # less the number of points of the most common label
    def loss_cons_1(model, k, t):
        return model.Loss[t] >= model.N[t] - model.NN[k, t] - model.n * (1 - model.c[k, t])

    model.LossConstraint1 = pyo.Constraint(model.K, model.tL,
                                           rule=loss_cons_1)

    def loss_cons_2(model, k, t):
        return model.Loss[t] <= model.N[t] - model.NN[k, t] + model.n * model.c[k, t]

    model.LossConstraint2 = pyo.Constraint(model.K, model.tL,
                                           rule=loss_cons_2)

    # Calculate the number of points of label k in leaf node t
    def calculate_point_cons_1(model, k, t):
        return model.NN[k, t] == 0.5 * sum((1 + model.y[i, k]) * model.z[i, t] for i in model.I)

    model.CalculatePointConstraint1 = pyo.Constraint(model.K, model.tL,
                                                     rule=calculate_point_cons_1)

    # Calculate the total number of points in leaf node t
    def calculate_point_cons_2(model, t):
        return model.N[t] == sum(model.z[i, t] for i in model.I)

    model.CalculatePointConstraint2 = pyo.Constraint(model.tL,
                                                     rule=calculate_point_cons_2)

    # Enforce a single class prediction at each leaf node t that contains points
    def enforce_cls_cons(model, t):
        return sum(model.c[k, t] for k in model.K) == model.l[t]

    model.EnforceClassConstraint = pyo.Constraint(model.tL,
                                                  rule=enforce_cls_cons)

    # Enfore the data point i to follow the splits that are required by the structure of the tree.
    def split_right_cons(model, i, t, m):
        # i in model.I, t and m in model.tL_AR pair which represent (leaf node, right ancestor node)
        # left hand side
        lhs = sum(model.X[i, j]*model.a[j, m] for j in model.J)
        # right hand side
        rhs = model.b[m] - (1-model.z[i, t])
        return lhs >= rhs

    model.SplitRightConstraint = pyo.Constraint(model.I, model.tL_AR,
                                                rule=split_right_cons)

    def split_left_cons(model, i, t, m):
        # i in model.I, t and m in model.tL_AL pair which represent (leaf node, left ancestor node)
        # left hand side
        lhs = sum(model.X[i, j]*model.a[j, m]
                  for j in model.J) + model.epsilon_min

        # lhs = sum((model.X[i, j] + model.epsilons[j])*model.a[j, m]
        #           for j in model.J)

        # right hand side
        rhs = model.b[m] + (1-model.z[i, t])*(1 + model.epsilon_max)
        return lhs <= rhs

    model.SplitLeftConstraint = pyo.Constraint(model.I, model.tL_AL,
                                               rule=split_left_cons)

    # Enforce each point i  be assigned to exactly one leaf node t
    def enforce_point_leaf_cons(model, i):
        return sum(model.z[i, t] for t in model.tL) == 1

    model.EnforcePointLeafNodesConstraint = pyo.Constraint(model.I,
                                                           rule=enforce_point_leaf_cons)

    # Enforce a minimum number of points at leaf node t
    def enforce_min_point_leaf_cons_1(model, i, t):
        return model.z[i, t] <= model.l[t]

    model.EnforceMinPointLeafNodesConstraint1 = pyo.Constraint(model.I, model.tL,
                                                               rule=enforce_min_point_leaf_cons_1)

    def enforce_min_point_leaf_cons_2(model, t):
        return sum(model.z[i, t] for i in model.I) >= model.min_samples_leaf * model.l[t]

    model.EnforceMinPointLeafNodesConstraint2 = pyo.Constraint(model.tL,
                                                               rule=enforce_min_point_leaf_cons_2)

    # Enforce leaf node l[tL] = 1 contain points if its parent branching node has split
    def enforce_min_point_leaf_cons_3(model, t, p):
        # t, p represent a pair of leaf nodes and their parent nodes: (leaf node, parent node)
        return model.l[t] >= model.d[p]

    model.EnforceMinPointLeafNodesConstraint3 = pyo.Constraint(model.tL_P,
                                                               rule=enforce_min_point_leaf_cons_3)

    # Enforce only 1 feature be splited if branch at node t
    # Also, enforce no feature be splited if no branching at node t
    def enfore_feature_branch_cons(model, t):
        return sum(model.a[j, t] for j in model.J) == model.d[t]

    model.EnforceFeatureBranchConstraint = pyo.Constraint(model.tB,
                                                          rule=enfore_feature_branch_cons)

    # Enforce split threshold is 0 if no branching at node t
    def enforce_threshold_branch_cons(model, t):
        return model.b[t] <= model.d[t]

    model.EnforceThresholdBranchConstraint = pyo.Constraint(model.tB,
                                                            rule=enforce_threshold_branch_cons)

    # Enforce no branching at node t if its parent node has no branching
    def enforce_parent_branch_cons(model, t, p):
        # t, p represent a pair of branch nodes and their parent nodes: (branch node, parent node)
        return model.d[t] <= model.d[p]

    model.EnforceParentBranchConstraint = pyo.Constraint(model.tB_P,
                                                         rule=enforce_parent_branch_cons)

    # # Enforce the root node must be a split:
    # def enforce_root_be_split_cons(model):
    #     return model.d[1] == 1

    # model.EnforceRootAsSplitConstraint = pyo.Constraint(
    #     rule=enforce_root_be_split_cons)

    # ======================================================================
    # ========================= SOLVE MILP MODEL ===========================
    # ======================================================================

    # ========================= Solve Model ================================
    # FIXME: Warm start
    # model.a[7, 1] = 1
    # model.a[7, 2] = 1
    # model.a[6, 3] = 1
    # model.l[16] = 1
    # model.l[23] = 1
    # model.l[24] = 1
    # # model.l[25] = 1
    # model.l[26] = 1
    # model.l[27] = 1
    # model.l[31] = 1

    # Fix d[1] == 1
    model.d[1].fix(1)

    # Fix at least two leaf nodes == 1
    most_left_leaf_node = _get_mostright_child(node=2,
                                               max_depth=max_depth)
    most_right_leaf_node = _get_mostright_child(node=3,
                                                max_depth=max_depth)

    model.l[most_left_leaf_node].fix(1)
    model.l[most_right_leaf_node].fix(1)

    solver_results = solver_configured.solve(model,
                                             tee=verbose
                                             #  warmstart=True
                                             )

    # ==================== Check Feasibility and Optimality and Resolve =======================
    # if the solver does not show termination condition as feasible or optimal
    # it will raise RuntimeError, then the algorithm will resolve the problem
    # by forcing the root node as a split
    try:
        _check_solver_termination(solver_results=solver_results)
    except RuntimeError:
        warnings.warn("No feasible solution obtained. "
                      " The algorithm will enforce to split at the node 1 and resolve the model."
                      )
        # Fix d[1] == 1 and Resolve the model
        model.d[1].fix(1)
        solver_results = solver_configured.solve(model, tee=verbose)
        _check_solver_termination(solver_results=solver_results)

    else:
        # Although feasible or optimal, if d[1] == 0 which means there is no splits,
        # fixing the root node has split : d[1] == 1
        if pyo.value(model.d[1]) == 0:

            warnings.warn("'alpha' complexity parameter maybe too large to "
                          " form a tree with valid splits."
                          " The algorithm will enforce to split at the node 1 and resolve the model."
                          )
            # Fix d[1] == 1 and Resolve the model
            model.d[1].fix(1)

            # Fix two leaf nodes == 1
            most_left_leaf_node = _get_mostright_child(node=2,
                                                       max_depth=max_depth)
            most_right_leaf_node = _get_mostright_child(node=3,
                                                        max_depth=max_depth)

            model.l[most_left_leaf_node].fix(1)
            model.l[most_right_leaf_node].fix(1)

            solver_results = solver_configured.solve(model, tee=verbose)
            _check_solver_termination(solver_results=solver_results)

    _check_solution(model_solved=model)

    return model


# %%
# The following functions are used to shape the model into a tree object

def _get_depth(node, max_depth):
    """Get the depth of a node index given 
    the maximum depth of a binary tree

    Parameters
    ----------
    node : int
        The node index of a binary tree. 
        NOTE: the index starts from 1

    max_depth : int
        The maximum depth of a binary tree

    Returns
    ----------
    depth : int
        The depth of the given node in the binary tree

    """
    if node == 1:
        return 0

    node_max_idx = math.pow(2, max_depth+1) - 1
    if node > node_max_idx:
        raise ValueError("'node' index out of range for a tree with"
                         " maximum depth {}. "
                         " 'node' must be in the range from 1 to {}"
                         .format(max_depth, int(node_max_idx))
                         )

    depth = 1
    while depth <= max_depth:
        most_left_node = math.pow(2, depth)
        most_right_node = math.pow(2, depth+1) - 1
        if node >= most_left_node and node <= most_right_node:
            break
        else:
            depth += 1

    return depth

# %%


def _get_child(node, direction):
    """Get the child index of a node by specifing the right or left
    in a full binary tree

    Parameters
    ----------
    node : int
        The node index of a binary tree. 
        NOTE: the index starts from 1

    direction : str
        "left" or "right"

    Returns
    ----------
    child : int
        The child node index

    """
    if node <= 0:
        raise ValueError("'node' index must be greater or equal to 1")

    all_directions = ["left", "right"]

    if direction not in all_directions:
        raise ValueError("'direction' must be either 'left' or 'right'.")

    if direction == "left":
        child = node * 2
    elif direction == "right":
        child = node * 2 + 1

    return child

# %%


def _get_mostright_child(node, max_depth):
    """Get the most right child index of a node, 
    given the maximum depth of a full binary tree.

    If the node is leaf, return itself.

    Parameters
    ----------
    node : int
        The node index of a binary tree. 
        NOTE: the index starts from 1

    max_depth : int
        The maximum depth of a binary tree

    Returns
    ----------
    node : int
        The most-right child index of the given node 

    """
    # the depth of the given node
    depth = _get_depth(node=node, max_depth=max_depth)

    diff = max_depth - depth

    while diff > 0:
        child_right = _get_child(node=node, direction="right")
        node = child_right
        diff -= 1

    return node

# %%


def _get_gini(value):
    """Calculate the Gini impurity

    Gini = sum[i] (x[i]/ sum(x[i]) * (1 - x[i]/ sum(x[i]) ))

    where x[i] represent the number of samples in class 'i'. 

    Parameters
    ----------
    value : np.ndarray of double, shape [max_n_classes]
        Contains the number of samples in each class at a certain node.

    Returns
    ----------
    gini : float
        Gini Impurity
    """
    sum_ = value.sum()

    if sum_ == 0:
        sum_ = 1

    prob = value/sum_
    gini = prob * (1-prob)
    gini = gini.sum()

    return gini


def _get_gini_array(value):
    """Calculate the Gini impurity

    Parameters
    ----------
    value : np.ndarray of double, shape [node_count, n_outputs, max_n_classes]
        Contains the number of samples in each class at each node.

    Returns
    ----------
    gini : np.ndarray of double, shape [node_count]
        Gini Impurity
    """
    n_outputs = value.shape[1]
    if n_outputs == 1:
        gini = np.apply_along_axis(_get_gini, axis=1, arr=value[:, 0, :])
    else:
        raise ValueError(
            "The current version does not support value with multi-outputs.")

    return gini

# %%


def _get_entropy(value):
    """Calculate the Entropy

    Entropy = sum[i] -(x[i]/ sum(x[i])) * log2(x[i]/ sum(x[i]))

    where x[i] represent the number of samples in class 'i'. 

    Parameters
    ----------
    value : np.ndarray of double, shape [max_n_classes]
        Contains the number of samples in each class at a certain node.

    Returns
    ----------
    entropy : float
        Entropy value
    """
    sum_ = value.sum()

    if sum_ == 0:
        sum_ = 1

    prob = value/sum_
    prob[prob == 0] = 1

    entropy = -1 * prob * np.log2(prob)
    entropy = entropy.sum()

    return entropy


def _get_entropy_array(value):
    """Calculate the Gini impurity

    Parameters
    ----------
    value : np.ndarray of double, shape [node_count, n_outputs, max_n_classes]
        Contains the number of samples in each class at each node.

    Returns
    ----------
    entropy : np.ndarray of double, shape [node_count]
        Gini Impurity
    """
    n_outputs = value.shape[1]
    if n_outputs == 1:
        entropy = np.apply_along_axis(_get_entropy, axis=1, arr=value[:, 0, :])
    else:
        raise ValueError(
            "The current version does not support value with multi-outputs.")

    return entropy
# %%


def _apply_split(X_transformed, y_transformed, feature, threshold, direction):
    """Apply a split to the training data and 
    return the number of samples in each class

    Parameters
    ----------

    X_transformed : nd-array of shape (n_samples, n_features)
        Transformed training data.

    y_transformed : nd-array of shape (n_samples, n_class)
        Transformed target data. Note that the -1 should convert to 0

    feature : int
        Indicate the feature index in X_transformed to be splited

    threshold : float
        The threshold value for the split

    direction : str, ["left", "right"]
        Indicate the left split or rigth split


    Returns
    ----------
    X_remain : nd-array of shape (n_samples, n_features)
        Transformed training data after spliting

    y_remain : nd-array of shape (n_samples, n_class)
        Transformed target data after spliting

    value : nd-array of double, shape (n_class)
        The number of samples in each class

    """

    if direction == "left":
        X_index = np.where(X_transformed[:, feature] < threshold)[0]
    elif direction == "right":
        X_index = np.where(X_transformed[:, feature] >= threshold)[0]

    X_remain = X_transformed[X_index, :]
    y_remain = y_transformed[X_index, :]

    value = np.apply_along_axis(np.sum, axis=0, arr=y_remain)

    return X_remain, y_remain, value

# %%


def _get_value(node, X, y, nodes_active, feature_, threshold_):
    """Get the number of samples in each class for node i

    Parameters
    ----------
    node : int
        The node label. Root node is 1.

    X : nd-array of shape (n_samples, n_features)
        Transformed training data

    y : nd-array of shape (n_samples, n_class)
        Transformed target data. Note that the -1 should convert to 0

    nodes_active : nd-array of shape (node_count, )
        The active nodes including branch nodes and leaf nodes

    feature_ : nd-array of shape (node_count, )
        Indicate the feature index in X_transformed to be splited at node i

    threshold_ : nd-array of shape (node_count, )
        The scaled threshold value for the split at node i

    Returns
    ----------
    value : nd-array of double, shape (n_class)
        The number of samples in each class
    """

    ancestors_left = _get_ancestors(node=node, kind="left")
    ancestors_right = _get_ancestors(node=node, kind="right")

    if not ancestors_left and not ancestors_right:
        value = np.apply_along_axis(np.sum, axis=0, arr=y)
        return value

    if ancestors_left:
        for ancestor in ancestors_left:
            if ancestor in nodes_active:
                ancestor_idx = nodes_active.index(ancestor)
                feature_idx = feature_[ancestor_idx]
                threshold_value = threshold_[ancestor_idx]
                X, y, value = _apply_split(X_transformed=X,
                                           y_transformed=y,
                                           feature=feature_idx,
                                           threshold=threshold_value,
                                           direction="left")
    if ancestors_right:
        for ancestor in ancestors_right:
            if ancestor in nodes_active:
                ancestor_idx = nodes_active.index(ancestor)
                feature_idx = feature_[ancestor_idx]
                threshold_value = threshold_[ancestor_idx]
                X, y, value = _apply_split(X_transformed=X,
                                           y_transformed=y,
                                           feature=feature_idx,
                                           threshold=threshold_value,
                                           direction="right")

    return value
# %%


class ModelTree:
    """Shape the fitted model to have array-based representaion of
    a binary decision tree.

    Parameters
    ----------
    model : a class object of pyomo.ConcreteModel
        The fitted Optimal Classification Tree model

    scaler_X: a class object of sklearn.preprocessing._data.MinMaxScaler
        The fitted scaler for the training data

    feature_removed_idx: np.array of int, shape [n_features_removed]
        The removed indexes of features

    criterion : str, default="gini"
        Available ["gini", "entropy"]
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    nodes_fashion : str, default="dfs"
        The orde of nodes by using Deep First Search ("dfs") or 
        Breath First Search ("bfs"). 
        **Don't use this parameter unless you know what you do**

    Attributes
    ----------
    node_count : int
        The number of nodes (branch nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    max_n_classes : int
        The maximum number of classes.

    n_classes : list
        The number of classes for each outputs

    n_outputs : int
        The number of outputs 

    n_features : int
        The number of features

    n_leaves : int
        The number of leaf nodes

    children_left : np.ndarray of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == -1. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : np.ndarray of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == -1. Otherwise,
        children_right[i] > i. This child handles the case where 
        X[:, feature[i]] > threshold[i].

    feature : np.ndarray of int, shape [node_count]
        feature[i] holds the feature to split on, for the branch node i.
        For leave nodes, feature[i] == -2.

    threshold : np.ndarray of double, shape [node_count]
        threshold[i] holds the threshold for the branch node i.
        For leave nodes, threshold[i] == -2.0

    value : np.ndarray of double, shape [node_count, n_outputs, max_n_classes]
        Contains the number of samples in each class at each node.
        This value will be used for making prediction and calculating impurity

    impurity : np.array of double, shape [node_count]
        impurity[i] holds the impurity measurement at node i.

        * Gini impurity will be calculated by specifying criterion="gini"
        * Entropy will be calculated by specifying criterion="entropy"

    n_node_samples : np.array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : np.array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.

    """

    def __init__(self,
                 model,
                 scaler_X,
                 feature_removed_idx,
                 classes,
                 criterion="gini",
                 nodes_fashion="dfs"):

        self.model = model
        self.scaler_X = scaler_X
        self.feature_removed_idx = feature_removed_idx
        self.classes = classes
        self.criterion = criterion
        self.nodes_fashion = nodes_fashion

    def shape(self, X_transformed, y_transformed):
        """Shape the array-based representation of a binary decision tree.

        The i-th element of each array holds information about the node 'i'.
        Node 0 is the tree's root.

        Parameters
        ----------
        X_transformed : nd-array of shape (n_samples, n_features)
            Transformed training data.

        y_transformed : nd-array of shape (n_samples, n_class)
            Transformed target data. Note that the -1 should convert to 0
        """
        all_fashion = ["dfs", "bfs"]
        if self.nodes_fashion not in all_fashion:
            raise ValueError("'nodes_fashion' must be either 'dfs' or 'bfs'.")

        all_criterion = ["gini", "entropy"]
        if self.criterion not in all_criterion:
            raise ValueError("'criterion' must be either 'gini' or 'entropy'.")

        # Get the maximum number of classes.
        model_K_set = list(self.model.K)
        self.max_n_classes = len(model_K_set)

        # Force to only predict one label
        self.n_outputs = 1
        self.n_classes = [self.max_n_classes]

        # Get the number of features (already remove useless features)
        model_J_set = list(self.model.J)
        self.n_features = len(model_J_set)

        # Get the node counts, capacity, number of leaves
        branch_nodes = list(self.model.tB)
        leaf_nodes = list(self.model.tL)

        branch_nodes_active = []
        leaf_nodes_active = []

        for node in branch_nodes:
            if self.model.d[node].value == 1:
                branch_nodes_active.append(node)

        for node in leaf_nodes:
            if self.model.l[node].value == 1:
                leaf_nodes_active.append(node)

        node_count = len(branch_nodes_active) + len(leaf_nodes_active)

        self.node_count = node_count
        self.capacity = node_count
        self.n_leaves = len(leaf_nodes_active)

        # Get the max_depth
        max_depth_org = int(math.log(leaf_nodes[0], 2))

        last_branch_node = branch_nodes_active[-1]

        depth_last_branch_node = _get_depth(node=last_branch_node,
                                            max_depth=max_depth_org)

        self.max_depth = depth_last_branch_node + 1

        # Get the Breath-First Order for active nodes
        if self.nodes_fashion == "bfs":
            nodes_active = []
            for node in branch_nodes_active:
                child_left = _get_child(node=node, direction="left")
                child_right = _get_child(node=node, direction="right")

                if node not in nodes_active:
                    nodes_active.append(node)

                if child_left not in nodes_active:
                    nodes_active.append(child_left)

                if child_right not in nodes_active:
                    nodes_active.append(child_right)

        # Get the Deep-First Order for active nodes
        elif self.nodes_fashion == "dfs":
            nodes_active = []
            queue = [branch_nodes_active[0]]
            while queue:
                node = queue.pop()
                if node not in nodes_active:
                    nodes_active.append(node)
                    if node in branch_nodes_active:
                        child_left = _get_child(node=node, direction="left")
                        child_right = _get_child(node=node, direction="right")
                        queue.append(child_right)
                        queue.append(child_left)
        # FIXME:
        # Check if the the tree is valid because sometimes there might exist
        # extreme small numerical issue in the OCT tree contraints such that
        # the number of nodes_active is not equal to the branch nodes + leaf nodes
        if len(nodes_active) != node_count:
            raise RuntimeError("Tree is not constructed properly. "
                               "Please check `model.SplitLeftConstraint` in solve_oct_MILP()")

        # Get left and right node index starting from 0
        #   children_left
        #   children_right
        children_left_ = np.zeros((self.node_count,),
                                  dtype=int)

        children_right_ = np.zeros((self.node_count,),
                                   dtype=int)

        for idx, node in enumerate(nodes_active):
            if node in branch_nodes_active:
                child_left = _get_child(node=node, direction="left")
                child_right = _get_child(node=node, direction="right")

                left_idx = nodes_active.index(child_left)
                right_idx = nodes_active.index(child_right)

                children_left_[idx] = left_idx
                children_right_[idx] = right_idx

            if node in leaf_nodes_active:
                children_left_[idx] = -1
                children_right_[idx] = -1

            if node not in branch_nodes_active and node not in leaf_nodes_active:
                # this node is fake leaf node, which will be fixed later
                children_left_[idx] = -1
                children_right_[idx] = -1

                # replace fake leaf nodes in nodes_active with the actual leaf nodes
                most_right_child = _get_mostright_child(node=node,
                                                        max_depth=max_depth_org)
                nodes_active[idx] = most_right_child

        self.children_left = children_left_
        self.children_right = children_right_
        self.nodes_active = nodes_active

        # Get several information on each node:
        #   features : which feature index to split at node i
        #   threshold : the threshold value to split for feature index at node i
        #   value: the number of samples in each class at node i
        #   n_node_samples: the number of training samples reaching node i.
        #   weighted_n_node_samples: the weighted number of training samples reaching node i.
        features_ = np.zeros((self.node_count,),
                             dtype=int)

        threshold_ = np.zeros((self.node_count,),
                              dtype=float)

        threshold_scaled_ = np.zeros((self.node_count,),
                                     dtype=float)

        value_ = np.zeros((self.node_count, self.n_outputs, self.max_n_classes),
                          dtype=float)

        n_node_samples_ = np.zeros((self.node_count,),
                                   dtype=int)

        weighted_n_node_samples_ = np.zeros((self.node_count,),
                                            dtype=float)

        for idx in range(self.node_count):
            node = nodes_active[idx]
            if node in branch_nodes_active:
                b_vector = np.zeros((1, self.n_features), dtype=float)
                for j in model_J_set:
                    if self.model.a[j, node].value == 1:
                        # feature index for splitting
                        features_[idx] = j-1
                        # threshold value
                        b_value = self.model.b[node].value
                        b_vector[0, j-1] = b_value
                        threshold_scaled_[idx] = b_value
                        # recover threshold to the original scale X
                        b_vector = self.scaler_X.inverse_transform(b_vector)
                        threshold_[idx] = b_vector[0, j-1]
                        break

            if node in leaf_nodes_active:
                features_[idx] = -2
                threshold_[idx] = -2.0
                threshold_scaled_[idx] = -2.0
                n_node_samples_[idx] = self.model.N[node].value
                weighted_n_node_samples_[idx] = self.model.N[node].value
                if self.n_outputs == 1:
                    for k_idx, k in enumerate(model_K_set):
                        value_[idx, 0, k_idx] = self.model.NN[k, node].value

        # convert -1 to be 0 in y_transformed
        y_transformed[y_transformed == -1] = 0
        # Recover value_, n_node_samples, weighted_n_node_samples for branch nodes
        for node in branch_nodes_active:
            idx = nodes_active.index(node)
            if self.n_outputs == 1:
                value_[idx, 0, :] = _get_value(node=node,
                                               X=X_transformed,
                                               y=y_transformed,
                                               nodes_active=nodes_active,
                                               feature_=features_,
                                               threshold_=threshold_scaled_)

                node_samples = value_[idx, 0, :].sum()
                n_node_samples_[idx] = node_samples
                weighted_n_node_samples_[idx] = node_samples

        # Get several information on each node:
        #   impurity: gini or entropy
        if self.criterion == "gini":
            impurity_ = _get_gini_array(value=value_)
        elif self.criterion == "entropy":
            impurity_ = _get_entropy_array(value=value_)

        self.feature = features_
        self.threshold = threshold_
        self.value = value_
        self.n_node_samples = n_node_samples_
        self.weighted_n_node_samples = weighted_n_node_samples_
        self.impurity = impurity_

        return(self)

    def predict(self, X, kind=None):
        """Finds the terminal region (e.g., leaf node `i`) for each sample in X
        and return the `value[i]` for node `i`.

        In other words, every sample in X would travese the Optimal Tree
        and stop at the a leaf node. The prediction would be the `value` information
        on that node. Recall that `value[i]` represents the number of training samples
        in each class in node `i`.  

        Parameters
        ----------
        X : nd-array of shape (n_samples, n_features)
            Data for prediction

        kind : str, default=None
            Used to output class label or probability or the number of samples in each class
            in the leaf node.

            By default, it will output `value`, i.e., the number of samples in each class in
            the leaf node.
            kind="class", predict the class label
            kind="prob", predict the probability that is the fraction of samples of the same
            class in a leaf.

        Returns
        ----------
        prediction : nd-array of shape (n_samples, n_outputs, n_classes)
            By default, it contains the value[i] for leaf node `i` where a sample stop at.
            By specifying 'kind', it could contain class label or probability

        """
        all_kind = ["class", "prob"]
        if kind and kind not in all_kind:
            raise ValueError(
                "'kind' must be either 'class' or 'prob' or None.")

        # Remove useless features if exist
        if self.feature_removed_idx.size > 0:
            X = np.delete(arr=X, obj=self.feature_removed_idx, axis=1)

        n_samples = X.shape[0]

        # Contains the leaf node for each sample in X
        samples_nodes = np.zeros((n_samples,), dtype=int)

        for i in range(n_samples):
            node_idx = 0
            # While not leaf node
            while self.children_left[node_idx] != -1:
                # get threshold
                threshold_value = self.threshold[node_idx]
                # get split feature index
                feature_idx = self.feature[node_idx]
                # left split
                if X[i, feature_idx] < threshold_value:
                    node_idx = self.children_left[node_idx]
                # right split
                else:
                    node_idx = self.children_right[node_idx]

            samples_nodes[i] = node_idx

        prediction = self.value.take(samples_nodes, axis=0, mode='clip')

        if self.n_outputs == 1:
            prediction = prediction.reshape(n_samples, self.max_n_classes)

        if kind:
            if kind == "class":
                return self.classes.take(np.argmax(prediction, axis=1), axis=0)

            elif kind == "prob":
                normalizer = prediction.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                prediction /= normalizer
                return prediction
        else:
            return prediction


# %%
def _check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    This utility is meant to be used internally by estimators themselves,
    typically in their own predict / transform methods.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))

    if msg is None:
        msg = ("This {} instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator."
               .format(type(estimator).__name__))

    if not hasattr(estimator, 'fit'):
        raise TypeError("{} is not an estimator instance.".format(estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator)
                 if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError(msg)
