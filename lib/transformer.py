import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from operator import itemgetter
import sklearn
from sklearn.preprocessing import PowerTransformer
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
set_config(transform_output="pandas")

class ShiftData(BaseEstimator, TransformerMixin):
    """ Transformer to shift the data by the mean of all values
    
    This transformer ist compatible with scikit-learn utilities.

    Parameters
    ----------
    x_init : np.ndarray, default=None
        x values which should be used to calculate the mean value used to shift the data
    
    x_shift : float, default=None
        value to shift the data, instead for mean value of x_init

    Source
    ------
    https://www.andrewvillazon.com/custom-scikit-learn-transformers/
    """
    def __init__(self, x_init: np.ndarray = None, x_shift: float = None) -> None:
        super().__init__()
        self.x_init = x_init
        self.x_shift = x_shift
        pass
    def fit(self, x, y=None):
        """ Calculates the shift value"""
        if self.x_shift == None:
            self.x_shift= np.mean(self.x_init)
        return self
    def transform(self, x, y=None):
        """ Transformes the data with the shift value"""
        x = x - self.x_shift
        return x
    
class SkewScaleTransformer(BaseEstimator, TransformerMixin):
    """ Transformer that performs log-transformation and standard-normal-distribution scaling
    for a single column containing negative and positive values.
    
    NOTE The transformer uses pandas DataFrames.

    To enable the compatibility with the scikit-learn library, transform_output should be set to "pandas".
    >>> from sklearn import set_config
    >>> set_config(transform_output="pandas")

    Attributes
    ----------
    scaler : object
        sklearn StandardScaler
    
    _sign : int
        used to retransform the data
    
    """
    def __init__(self) -> None:
        super().__init__()
        self.scaler = StandardScaler()

    def fit(self, x=None):
        """ Fits the transformer to column data

        To retransform the data, the median sign of the column is stored, since
        log-transformation of negative values is not possible.

        Parameters
        ----------
        x : array-like
            column data
        """
        self._sign = np.median(np.sign(x))
        x_trans = np.log(np.absolute(x))
        self.scaler.fit(x_trans)
        return self
    
    def transform(self, x=None):
        """ Transforms the column data

        Parameters
        ----------
        x : array-like
            column data
        """
        x_trans = np.log(np.absolute(x))
        x_trans_scale = self.scaler.transform(x_trans)
        return x_trans_scale
    
    def inverse_transform(self, x=None):
        """ Retransforms the data
        
        Parameters
        ----------
        x : array-like
            column data
        """
        x_trans_scale = x
        columns = x.columns
        x_trans = self.scaler.inverse_transform(x_trans_scale)
        x = np.exp(x_trans)*self._sign
        x = pd.DataFrame(data=x, columns=columns)
        return x

class Selector(BaseEstimator, TransformerMixin):
    """ Transformer that performs hierarchical or bidirectional feature selection with
    consequently elimination of non significant coefficients.
    
    Hierarchical feature seletion is recommended for polynomial feature selection. Hereby
    the selector adds polynomial features in an ascending order starting from the intercept or
    a predefined minimum degree. While checking the significance of coefficients, only the highest
    coefficient is checked and eventually eliminated to sustain the structure of hierarchical
    polynoms. The model is recalculated after each elimination.

    For bidirectional feature selection, features from a feature subset are added with a forward
    selection step and removed during a backward selection step. The procedures continues in a loop 
    until no better subset is found. Afterwards, non significant coefficients will be gradually
    omited by descending order of their p-value, with recalculation of the model after each
    elimination.

    NOTE The transformer uses pandas DataFrames.

    To enable the compatibility with the scikit-learn library, transform_output should be set to "pandas".
    >>> from sklearn import set_config
    >>> set_config(transform_output="pandas")

    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator
    
    method : {"hierarchical", "bidirectional"}, default="hierarchical"
        Wheter to perform hierarchcial or bidirectional selection
    
    scoring : str, default = None
        A single str to evaluate the predictions on the data
        If None, the estimator's default score method is used
    
    deg_min : int, default = 0
        Minimum polynomial degree to restrict the hierarchical feature selection
        For bidirectional selection, manual settings of deg_min will be ignored.
    
    sign_level : float, default = 0.05
        Significant level to eliminate non significant coefficients
    
    DEBUG : bool, default = True
        If True, debug messages will be printed

    Attributes
    ----------
    summary_step_restults : pd.DataFrame
        Chronological list of each step of the feature selection process.
    best_coeff_names : list
        Name of the best coefficients
    best_coeff_values : list
        Coefficient values corresponding to best_coeff_names
    best_score : float
        Score of the best feature subset
    best_pvalues : list
        p-values corresponding to best_coeff_names

    Source
    ------
    Manual p-value calculation: https://www.datacourses.com/find-p-value-significance-in-scikit-learn-3810/
    
    """
    def __init__(self, estimator, method: str = "hierarchical", scoring: str = None, deg_min: int = 0, sign_level: float = 0.05, DEBUG=True) -> None:
        self.estimator = estimator
        self.method = method
        self.scoring = scoring
        if scoring != None:
            self.estimator.set_params(**{"scoring": str(self.scoring)})
        else: self.scoring = "score"
        if self.method == "hierarchical":
            self.deg_min = deg_min
        if self.method == "bidirectional":
            self.deg_min = 0
        self.sign_level = sign_level
        self.DEBUG = DEBUG
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame =None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        y : pd.DataFrame
            Target values

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X.rename(columns={"1" : "intercept"}, inplace=True)
        included = list(X.columns)[:(self.deg_min+1)]
        # initial fit
        fit_result = self._get_fit_results(X[included], y)
        self._update_summary_step_results(fit_result, "selection")
        self._update_best_results(fit_result)
        # feature selection
        while True:
            changed = False
            # forward step
            best_col_temp = ""
            excluded = list(set(X.columns)-set(included))
            excluded.sort()
            if len(excluded) == 0: break
            for col in excluded:
                included.append(col)
                included.sort()
                if included in list(self._summary_step_results_intern["coeff_names"]):
                    if self.method == "bidirectional":
                        included.remove(col)                    
                    continue
                fit_result = self._get_fit_results(X[included], y)
                self._update_summary_step_results(fit_result, "selection")
                if (np.isnan(self.best_score)==True and np.isnan(fit_result[str(self.scoring)])==False) or fit_result[str(self.scoring)] > self.best_score:
                    changed = True
                    self._update_best_results(fit_result)
                    best_col_temp = col
                if self.method == "hierarchical":
                    #if not changed: break # check all polynomials until given deg_max
                    changed = False
                if self.method == "bidirectional":
                    included.remove(col)
            # backward step
            if self.method == "bidirectional":
                if len(best_col_temp) > 0: included.append(best_col_temp)
                worst_col_temp = ""
                for col in included:
                    if len(included) > 2:
                        included.remove(col)
                        included.sort()
                        if included in list(self._summary_step_results_intern["coeff_names"]):
                            included.append(col)
                            continue
                        fit_result = self._get_fit_results(X[included], y)
                        self._update_summary_step_results(fit_result, "selection")
                        if (np.isnan(self.best_score)==True and np.isnan(self.best_score)==False) or fit_result[str(self.scoring)] > self.best_score:
                            changed = True
                            self._update_best_results(fit_result)
                            worst_col_temp = col
                        included.append(col)
                if len(worst_col_temp) > 0: included.remove(worst_col_temp)
            if not changed: break
        # remove non significant coefficients
        if self.best_pvalues != None:
            while True:
                if self.method == "hierarchical":
                    max_pval_idx = -1
                if self.method == "bidirectional":
                    max_pval_idx = self.best_pvalues.index(max(self.best_pvalues))
                if self.best_pvalues[max_pval_idx] <= self.sign_level:
                    break
                else:
                    included = self.best_coeff_names
                    included_temp = included.copy() # use temp copy otherwise self._summary_step_results_intern will be changed as well -> did not find better solution yet
                    del included_temp[max_pval_idx]
                    included = included_temp
                    if len(included) <= (self.deg_min+1): # add 1 due to intercept
                        if self.DEBUG == True:
                            print("Fit with significant coefficients not possible")
                        #fit_result = self._get_fit_results(X[[]], y)
                        fit_result = dict(zip(self._fit_result_keys, [["intercept"], [0], 0, [0]]))
                        self._update_summary_step_results(fit_result, "optimisation")
                        self._update_best_results(fit_result)
                        break
                    if included in list(self._summary_step_results_intern["coeff_names"]):
                        included_index = list(self._summary_step_results_intern["coeff_names"]).index(included)
                        temp_result = self._summary_step_results_intern.loc[included_index,:].to_dict()
                        self._update_best_results(temp_result)
                    else:
                        fit_result = self._get_fit_results(X[included], y)
                        self._update_summary_step_results(fit_result, "optimisation")
                        self._update_best_results(fit_result)         
        return self

    def _get_fit_results(self, X: pd.DataFrame, y: pd.DataFrame=None) -> dict:
        # Performs the estimator fit with X and y and returns a dict
        # containing coeff_names, coeff_values, score, pvalues of the fit.
        self.estimator.fit(X, y)
        coeff_names = list(X.columns)
        coeff_values = self.estimator.coef_
        if type(coeff_values) == np.ndarray:
            if coeff_values.shape[0] == 1:
                coeff_values = list(np.reshape(coeff_values, (coeff_values.shape[1])))
        else:
            coeff_values = list(coeff_values)
        score = self.estimator.score(X, y)
        if hasattr(self.estimator, "pvalues_"): pvalues = list(self.estimator.pvalues_)
        else:
            # calculate p values manually if not provided by estimator
            y_pred = self.estimator.predict(X)
            y_pred = np.reshape(y_pred, (y_pred.shape[0]))
            X_arr, y_arr = X.to_numpy(), y.to_numpy()
            y_arr = np.reshape(y_arr, (y_arr.shape[0]))
            y_mean = np.mean(y_arr)
            p = len(X_arr[0])-1 # len(X_arr[0]) includes intercept
            N = len(X_arr)
            M_S_R = (sum((y_arr-y_mean)**2))/p
            M_S_E = (sum((y_arr-y_pred)**2))/(N-p-1)
            v_b =  M_S_E*(np.linalg.inv(np.dot(X_arr.T,X_arr)).diagonal())
            s_b = np.sqrt(abs(v_b))
            t_b = np.array(coeff_values)/ s_b
            pvalues =[2*(1-stats.t.cdf(np.abs(i),(len(X_arr)-len(X_arr[0])))) for i in t_b]
        self._fit_result_keys = ["coeff_names", "coeff_values", str(self.scoring), "pvalues"]
        if "intercept" in coeff_names:
            idx = coeff_names.index("intercept")
            coeff_names_idx = coeff_names[idx]
            coeff_values_idx = coeff_values[idx]
            coeff_names.remove(coeff_names_idx)
            coeff_names.insert(0, coeff_names_idx)
            coeff_values.remove(coeff_values_idx)
            coeff_values.insert(0, coeff_values_idx)
        values = [coeff_names, coeff_values, score, pvalues]
        result_dict = dict(zip(self._fit_result_keys, values))
        return result_dict

    def _update_summary_step_results(self, fit_result: dict, step):
        # Initializes and updates summary_step_results with the result of _get_fit_results 
        if not hasattr(self, "summary_step_results"):
            self.summary_step_results = pd.DataFrame(None, columns=self._fit_result_keys)
            self._summary_step_results_intern = self.summary_step_results.copy()
        fit_result["step"] = step
        fit_result_formated ={}
        for key, value in fit_result.items():
            if (type(value) in [list, np.ndarray]) and all(isinstance(item, str) for item in value) == False:
                value = ["%.2e" % item for item in value]
            if type(value) == np.float64:
                value = "%.2f" % value
            fit_result_formated[key] = str(value).replace("[","").replace("]","").replace("(","").replace(")","").replace("array","")
        self._summary_step_results_intern = pd.concat([self._summary_step_results_intern, pd.DataFrame([fit_result])], ignore_index=True)
        self.summary_step_results = pd.concat([self.summary_step_results, pd.DataFrame(fit_result_formated, index=[0])], ignore_index=True)
        return
    
    def _update_best_results(self, fit_result):
        # updates and stores the variables of (current) best_coeff_names, best_coeff_values,
        # best_score and best_pvalues as class attributes
        self.best_coeff_names, self.best_coeff_values, self.best_score, self.best_pvalues = itemgetter(*self._fit_result_keys)(fit_result)
        return
    
    def transform(self, X, y=None):
        '''Selects the best feature subset after fit

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix prior selection

        Return
        ------
        X : pd.DataFrame
            Feature matrix made up of selected features
        '''
        X.rename(columns={"1" : "intercept"}, inplace=True)
        X = X[self.best_coeff_names]
        return X