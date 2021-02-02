
"""
This module gathers Large Scale Optimal Tree methods, including
optimal classification tree and optimal classification tree with hyperplanes
"""

# Authors: Enhao Liu<liu.5045@osu.edu>
# Theodore Allen<allen.515@osu.edu>
#
# License: ZZZZZ


import numpy as np
import pandas as pd
import numbers
import math
import warnings
from sklearn.utils import check_array
# from sklearn.utils.validation import check_is_fitted
from ._base import solve_oct_MILP, ModelTree
from ._base import _check_preprocess_X_y, _check_parameters, _check_configure_solver
from ._base import _check_is_fitted

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# __all__ = ["OptimalTreeClassifier"]

# %%


class OptimalTreeClassifier:
    """An Optimal Tree Classifier.

    Parameters
    ----------
    max_depth : int, default=2
        The maximum depth of the tree. 

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

    alpha : float, default=0.01
        The complexity parameter. Any number from 0.0 to 1.0. 
        If it becomes larger, it will result in less splits.

    criterion : str, default="gini"
        The function to measure the quality of a split. 
        Supported criteria are:
            - "gini" for the Gini impurity
            - "entropy" for the information gain.

        NOTE: This parameter has NO contribution on solving 
        the Optimal Classification Tree. Tuning this parameter would not
        improve the performance of the model. 
        It is used to re-calculate the "gini" or "entropy" for each node 
        in the solved tree for evaluating the feature importance.

    solver : str, default="gurobi"
        The optimization solver to use.
        Supported solvers are ["cplex", "gurobi", "glpk", "cbc"]

        NOTE: The most efficient solvers are "gurobi" and "cplex"
        "glpk" and "cbc" solovers might not be able to deliver feasible
        solutions in a short time.

        - For "gurobi" solver, it would invoke Gurobi Interactive Shell
        - For "cplex" solver, it would invoke IBM ILOG CPLEX Interactive Optimizer
        - For "glpk" solver, it would invoke GLPSOL, the GLPK command-line solver
        - For "cbc" solver, it would invoke CBC MILP Solver

    time_limit : int, default=5
        The time limit in minutes of running solver. Default is 5 minutes.
        It is used to early stop the solver.

    verbose : boolean, default=False
        If True, display the outputs of the solver.

    solver_options : dict, default=None
        Use to specify additonal configurations in solver to improve the 
        quality of Optimal Classification Tree.

        - default=None means that using the solver's default configurations.

        NOTE: Each solver has its own syntax for additional configurations. 
        See References for more details.
        -------------------------------------------------------------------
        * solver="gurobi", Available options [1][2] are:

            {"mip_cuts" : str}
                Global cut control setting in Gurobi. Available values are:

                - "auto" (default) : automatically generate cuts

                - "off" : shut off cuts

                - "moderate" : moderate cut generation

                - "aggressive" : aggressive cut generation

            {"mip_gap_tol" : float} default=1e-4
                Relative MIP gap tolerance. Any number from 0.0 to 1.0. 
                For example, mip_gap_tol=0.05 is to enable solver to stop as soon as
                it has found a feasible integer solution proved to be within
                five percent of optimal.

            {"mip_focus" : str}
                This parameter allows you to modify your high-level solution strategy, 
                depending on your goals. Available values are:

                - "balance" (default) : By default, the Gurobi solver strikes a balance between finding 
                            new feasible solutions and proving that the current solution is optimal.

                - "feasible" : More interested in finding feasible solutions quickly.
                               User may want a greater emphasis on feasibility and less emphasis on analysis and 
                               proof of optimality.

                - "optimal" : If you believe the solver is having no trouble finding good quality solutions, 
                              and wish to focus more attention on proving optimality.

                - "bound" : If the best objective bound is moving very slowly (or not at all), 
                            you may want to try to focus on the bound.


        * solver="cplex", Available options [3] are:

            {"mip_cuts" : str}
                Global cut control setting in Cplex. Available values are:

                - "auto" (default) : automatically generate cuts

                - "off" : shut off cuts

                - "moderate" : moderate cut generation

                - "aggressive" : aggressive cut generation

            {"mip_gap_tol" : float} default=1e-4
                Relative MIP gap tolerance. Any number from 0.0 to 1.0. 
                For example, mip_gap_tol=0.05 is to enable solver to stop as soon as
                it has found a feasible integer solution proved to be within
                five percent of optimal.

            {"mip_focus" : str}
                This parameter allows you to modify your high-level solution strategy, 
                depending on your goals. Available values are:

                - "balance" (default) : By default, the Cplex solver strikes a balance between finding 
                            new feasible solutions and proving that the current solution is optimal.

                - "feasible" : More interested in finding feasible solutions quickly.
                               User may want a greater emphasis on feasibility and less emphasis on analysis and 
                               proof of optimality.

                - "optimal" : If you believe the solver is having no trouble finding good quality solutions, 
                              and wish to focus more attention on proving optimality.

                - "bound" : If the best objective bound is moving very slowly (or not at all), 
                            you may want to try to focus on the bound.

                - "hidden" : This choice is intended for use on difficult models where 
                             a proof of optimality is unlikely, and where mip_focus="feasible" 
                             does not deliver solutions of an appropriately high quality.

        * solver="glpk", Available options [4] are:

            {"mip_cuts" : str or list}
                The cutting planes are generated by specifying cuts' names or strategies.
                Support cuts are in ["gomory", "mir", "cover", "clique"]. Available values are: 

                - None (default): no cuts 

                - "all" : adding all available cuts

                - list : contains the cuts' names. For example:
                       ["gomory"] : adding only Gomory cuts
                       ["gomory", "mir"] : adding both Gomory and MIR cuts

            {"mip_gap_tol" : float}
                Relative MIP gap tolerance. Any number from 0.0 to 1.0.
                For example, mip_gap_tol=0.05 is to enable solver to stop as soon as
                it has found a feasible integer solution proved to be within
                five percent of optimal.

            {"fp_heur" : boolean}
                Chooses whether or not to apply the Feasibility Pump heuristic on
                finding a feasible solution. It might help to obtain a solution quickly.

                - False (default) : the GLPK solver would not apply the heuristic method
                        to find a feasible solution.

                - True : Allow the solver to apply the heuristic method to find a feasible solution.

            {"backtrack" : str}
                The backtracking setting in GLPK solver. Available values:

                - "bestb" (default) : backtrack using node with best local bound

                - "dfs" : backtrack using depth first search

                - "bfs" : backtrack using breadth first search

                - "bestp" : backtrack using the best projection heuristic

        * solver="cbc", Available options [5] are:

            {"mip_cuts" : XXXXX}
                The cutting planes are generated by specifying cuts' names or strategies.
                Support cuts are in ["gomory", "mir", "cover", "clique"]. Available values are: 

                TODO: the current version does not allow configurations for "cbc" solver


    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    n_classes_ : int
        The number of classes (for single output problems)

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : ModelTree
        The underlying ModelTree object. Please refer to
        ``help(lsopt._base.ModelTree)`` for attributes of ModelTree object 

    TODO : the current version does not compute feature_importances_

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    References
    ----------

    .. [1] Gurobi: Documentation about Parameters
           https://www.gurobi.com/documentation/9.0/refman/parameters.html#sec:Parameters

    .. [2] Gurobi: Documentation about MIP Model
           https://www.gurobi.com/documentation/9.0/refman/mip_models.html

    .. [3] Cplex: Documentation about Parameters
           https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/InteractiveOptimizer/topics/commands.html

    .. [4] GLPK: Using GLPSOL, the GLPK command-line solver
           https://en.wikibooks.org/wiki/GLPK/Using_GLPSOL

    .. [5] CBC : TODO: add references for cbc solver
    """

    def __init__(self,
                 max_depth=2,
                 min_samples_leaf=1,
                 alpha=0.01,
                 criterion="gini",
                 solver="gurobi",
                 time_limit=5,
                 verbose=False,
                 solver_options=None):

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.criterion = criterion
        self.solver = solver
        self.time_limit = time_limit
        self.verbose = verbose
        self.solver_options = solver_options

        # Check parameters
        _check_parameters(alpha=self.alpha,
                          max_depth=self.max_depth,
                          min_samples_leaf=self.min_samples_leaf,
                          criterion=self.criterion)

        # Check solver and solver_option
        self.solver_options['time_limit'] = self.time_limit

        _check_configure_solver(solver=self.solver,
                                return_config=False,
                                **self.solver_options)

    def fit(self, X, y, sample_weight=None):
        """Build a Optimal Decision Tree Classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        TODO: the current version does not support sample weights.

        Returns
        -------
        self : OptimalTreeClassifier
            Fitted estimator.
        """

        # Check and preprocess X, y
        X_transformed, y_transformed, class_names, \
            L_hat, epsilons, scaler_X, feature_removed_idx =\
            _check_preprocess_X_y(X=X, y=y)

        self.classes_ = class_names

        # Solve Optimal Classification Tree
        model = solve_oct_MILP(X_transformed=X_transformed,
                               y_transformed=y_transformed,
                               L_hat=L_hat,
                               epsilons=epsilons,
                               alpha=self.alpha,
                               max_depth=self.max_depth,
                               min_samples_leaf=self.min_samples_leaf,
                               solver=self.solver,
                               verbose=self.verbose,
                               **self.solver_options)

        # Construct ModelTree object
        tree_ = ModelTree(model=model,
                          scaler_X=scaler_X,
                          feature_removed_idx=feature_removed_idx,
                          classes=self.classes_,
                          criterion=self.criterion,
                          nodes_fashion='dfs')

        tree_.shape(X_transformed=X_transformed,
                    y_transformed=y_transformed)

        self.tree_ = tree_

        self.n_classes_ = tree_.max_n_classes
        self.n_features_ = tree_.n_features
        self.n_outputs_ = tree_.n_outputs
        # self.feature_importances_ =
        # self.max_features_ =

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype="numeric", ensure_2d=True)

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {} and "
                             "input n_features is {} ".format(self.n_features_, n_features))

        return X

    def predict(self, X, check_input=True):
        """Predict class value for X.

        For a classification model, the predicted class for each sample in X is
        returned. 

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples. 

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : ndarray of shape (n_samples,) 
            The predicted classes
        """
        _check_is_fitted(self)

        X = self._validate_X_predict(X, check_input)

        pred = self.tree_.predict(X=X, kind='class')

        return pred

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples. 

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) 
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """

        _check_is_fitted(self)

        X = self._validate_X_predict(X, check_input)

        pred_prob = self.tree_.predict(X=X, kind='prob')

        return pred_prob
