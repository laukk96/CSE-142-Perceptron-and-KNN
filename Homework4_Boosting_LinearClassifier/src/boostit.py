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
        
        
        self.m = []
        
        pass
    
    def base_learning_algorithm(self, error_t):
        result = np.log((1-error_t)/error_t)
        return result
    
    def linear_classifier(self, X, y, weights):
        """
        Trains a linear classifier on the given data and weights.
        Returns: (weights, error)
        """
        m = self.n_samples
        n = self.n_features
        X = np.c_[X, np.ones(m)]  # Add bias term

        # Calculate weighted least squares solution
        
        W = np.diag(weights)
        inverse = np.linalg.inv(X.T.dot(W).dot(X))
        w = inverse.dot(X.T).dot(W).dot(y)

        # Calculate weighted error
        pred = X.dot(w)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        error = np.sum(weights[y != pred])
        
        print("Linear Classifier: Preds Size = ", pred.size, w)

        return w, error
        
        
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
        
        
        
        # start with uniform weights
        self.weights = np.full(n_samples, 1.0/n_samples)
        
        new_weights, error = self.linear_classifier(X, y, weights=self.weights)
        
        print(new_weights.shape, error)
        
        # reiterate on the classifier T times
        for i in range(self.T):
            # calculate predictions
            
            pass

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

