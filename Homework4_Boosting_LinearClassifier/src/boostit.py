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
        
        self.m_t = None
        self.w = None
        
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
        
        # Calculate weighted least squares solution
        W = np.diag(weights)
        XTW = X.T.dot(W)
        XTW_X = XTW.dot(X)
        inverse = np.linalg.pinv(XTW_X)  # Use pseudo-inverse for better numerical stability
        w = inverse.dot(XTW).dot(y)
        
        # set the class in the original
        self.w = w

        # Calculate predictions and weighted error
        # print(f'LINEAR CLASSIFER: -> X.shape = {X.shape}, w.shape = {w.shape}')
        pred = X.dot(w)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        error = np.sum(weights[y != pred])
        
        false_predictions = []
        true_predictions = []
        
        
        # for loop to create a list of true predictions / false predictions to return to fit()
        for i in range(self.n_samples):
            if y[i] == pred[i]:
                true_predictions.append(i)
            else:
                false_predictions.append(i)
        
        # # print(f'CLASSIFIER: -> true predictions: ({len(true_predictions)}) {true_predictions}')
        # print(f'CLASSIFIER: -> false predictions: ({len(false_predictions)}) {false_predictions}')
        # print(f'CLASSIFIER: -> W result: {w}, length: {w.shape}')

        # print("CLASSIFIER: -> Preds Size = ", pred.size, w)
        
        return w, error, true_predictions, false_predictions
        
        
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
        
        
        # reiterate on the classifier T times
        for i in range(self.T):
            # calculate predictions
            m_t, error_t, true_preds, false_preds = self.linear_classifier(X, y, weights=self.weights)
            self.m.append(m_t)
            self.m_t = m_t
            # print(f"T = {i} | fit(): model / error: ", m_t.shape, error_t)
            
            if error_t >= 1/2:
                break
            
            a_t = self.base_learning_algorithm(error_t=error_t)
            
            print("BEFORE: -> ", self.m_t)
            # self.m_t = self.m_t.dot(a_t)
            self.w = self.w.dot(a_t)
            
            print("AFTER: -> ", self.m_t)
            
            # update weights for 
            for j in range(len(false_preds)):
                index = false_preds[j]
                self.weights[index] = self.weights[index]/(2*error_t)
            
            # update weights correctly classified instances
            for j in range(len(true_preds)):
                index = true_preds[i]
                self.weights[index] = self.weights[index]/(2*(1-error_t))
            
            
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
        
        
        X = np.c_[X, np.ones(self.n_samples)]
        # print(f'PREDICT: -> X.shape = {X.shape}, self.w.shape = {self.w.shape}')
        pred = X.dot(self.w)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        
        return pred

