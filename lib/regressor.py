import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import statsmodels.api as sma

class smaWLS(BaseEstimator, RegressorMixin):
    """
    Wrapper for statsmodels weighted least squares Regression.

    The estimator fits a linear model with a higher weight for the first and
    last three datapoints. The scoring methods are implemented by the
    principle "higher is better", so negative values for AIC, BIC, RMSE are
    used.

    The estimator is compatible with the scikit-learn library. 

    Parameters
    ----------
    endpoint_weight : float, default = 1
        Weight for the first and last datapoints
        Endpoint_weight >= 10000 equals a force through the endpoints
    
    scoring : {"fvalue", "naic", "nbic","nrmse"}, default = "fvalue"
        A single str to evaluate the predictions on the data
    
    Attributes
    ----------
    results_ : object
        Fitted statsmodels object.
    
    coef_: list
        Estimated coefficients for the linear regression problem.
    
    pvalues_ : list
        p-values of the estimated coefficients
    
    Source
    -----
    https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible  
    """
    def __init__(self, endpoint_weight: float = 1, scoring: str = "fvalue") -> None:
        super().__init__()
        self.endpoint_weight = endpoint_weight
        self.scoring = scoring
    
    def fit(self, X, y):
        """ Fit linear model.
        
        Parameters
        ----------
        X : array-like
            Feature matrix (1D or 2D)

        y : array-like
            Target values
        
        Return
        ------
        self : object
            Fitted Estimator.
        
        """
        normal_weight = 1
        w = np.full((len(X),),normal_weight)
        w[:3], w[-3:] = self.endpoint_weight, self.endpoint_weight
        model_ = sma.WLS(y, X, np.sqrt(w))
        self.results_ = model_.fit()
        self.coef_ = self.results_.params
        self.pvalues_ = self.results_.pvalues
        return self
    
    def predict(self, X):
        """ Predict using the linear model.
        
        Parameters
        ----------
        X : array-like
            Feature matrix (1D or 2D)
        
        Returns
        -------
        C : array
            Returns predicted values.
        """
        return self.results_.predict(X)
    
    def score(self, X=None, y=None):
        """ Gets score of the fitted model
        
        If scoring method of estimator is set to "nrmse" the input values 
        for X and y will be used otherwise the score of the fit will be derieved.

        Parameters
        ----------
        X : array-like
            Feature matrix (1D or 2D)

        y : array-like
            Target values

        Return
        ------
        score : float
            score depending on the set scoring method of the estimator
        """
        if self.scoring in ["naic", "nbic"]:
            score = -getattr(self.results_,self.scoring[1:])
        elif self.scoring == "nrmse":
            score = -mean_squared_error(y_true=y, y_pred=self.results_.predict(X),squared=False)
        else: score = getattr(self.results_,self.scoring)
        return score
    
    def summary(self):
        """ Prints the summary of the fitted statsmodel"""
        print(self.results_.summary())
    
    def set_params(self, **params):
        '''Enables to set parameters      
        '''
        if not params:
            return self
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self