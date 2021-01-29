class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from lsopt.tree import OptimalTreeClassifier
    >>> from lsopt._exceptions import NotFittedError
    >>> try:
    ...     OptimalTreeClassifier().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This OptimalTreeClassifier instance is not fitted yet. Call 'fit' with
    appropriate arguments before using this estimator.")

    """
