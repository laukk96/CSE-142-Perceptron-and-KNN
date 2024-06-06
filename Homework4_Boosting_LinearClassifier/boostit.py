import numpy as np

class BoostingClassifier:
    """ Boosting for binary classification.
    Please build an boosting model by yourself.

    Examples:
    The following example shows how your boosting classifier will be used for evaluation.
    >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    >>> X_test, y_test = load_test_dataset()
    >>> clf = BoostingClassifier().fit(X_train, y_train)
    >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
    >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.

    """
    def __init__(self):
        # initialize the parameters here
        self.X = None
        self.Y = None
        self.T = 10
        
        pass
    

    def fit(self, X, y):
        """ Fit the boosting model.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            The input samples with dtype=np.float32.
        
        y : { numpy.ndarray } of shape (n_samples,)
            Target values. By default, the labels will be in {-1, +1}.

        Returns
        -------
        self : object
        """
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features
        
        self.weights = np.full(n_samples, 1.0/n_samples)
        
        print(self.weights)

        return self

    def predict(self, X): 
        """ Predict binary class for X.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples)
                 In this sample submission file, we generate all ones predictions.
        """
        return np.ones(X.shape[0], dtype=int)

