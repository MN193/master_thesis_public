import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
import pickle
from tqdm import tqdm
import itertools
import importlib
import lib.transformer as mytrans
import lib.regressor as myreg
import lib.utils as myutils
importlib.reload(myreg)
importlib.reload(mytrans)
importlib.reload(myutils)


# used to translate into supscrit and subscript numbers
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUB = str.maketrans("0123456789","₀₁₂₃₄₅₆₇₈₉")

class Baseline:
    """ A class to represent the baseline of a dataset throughout modelation.

    A Baseline is made up of coefficients which needs to be modelled seperately.
    E.g.: the baseline polynom is given by p(gate delay, laser energy, lambda) = a_0(GD, LE) + a_1(GD, LE)*lambda + ... + a_i(GD, LE)*lambda^i
          a coefficient model is made up of a_i(GD, LE) = b0 + b1*GD + b2*LE + b3*GD^2 + b4*LE^2 + b5*LE*GD
          
    The subclass ``Baseline.Model`` fits and predicts the entire baseline model to return
    [a_0, a_1, ... , a_i] for a given gate delay and laser energy. 
    Prior modelling, all empty replicate measurements will be discarded.  

    Parameters
    ----------
    dataset_name : str
        name of the dataset

    regression_coeff_df : pd.DataFrame
        dataframe with following column structure
        x_shift | intercept | x | ... | x^deg_max | degree | laser energy | gate delay

    regression_settings : dict
        dictionary containing the baseline settings used throughout baseline regression

    median : bool, default=True
        if True, only the median values for each condition (laser energy, gate delay) will be used
        throughout modelling
    
    Attributes
    ----------
    coeff_name_list : list
        formatted name of coefficients e.g. [a₀, a₁, a₂, a₃]
    
    model : object
        model of the baseline
    """
    def __init__(self, dataset_name: str, regression_coeff_df: pd.DataFrame, regression_settings: dict, median: bool = True) -> None:
        self.dataset_name = dataset_name
        self.median = median
        regression_coeff_df = regression_coeff_df[np.all(regression_coeff_df.iloc[:,1:-3]!=0, axis=1)]
        self.x_shift = np.median(regression_coeff_df[["x_shift"]])
        self.le_list = np.unique(regression_coeff_df[["laser energy"]])
        self.gd_list = np.unique(regression_coeff_df[["gate delay"]])
        if self.median==False:
            self.regression_coeff_df = regression_coeff_df
        if self.median==True:
            median_regression_coeff_df = pd.DataFrame(None, columns=regression_coeff_df.columns) 
            le_gd_comb = list(itertools.product(self.le_list, self.gd_list))
            for elem in le_gd_comb:
                le, gd = elem[0], elem[1]
                if np.any(regression_coeff_df[(regression_coeff_df["laser energy"]==le) 
                                              & (regression_coeff_df["gate delay"]==gd)]) == True:
                    append = pd.DataFrame(data=[np.median(regression_coeff_df[(regression_coeff_df["laser energy"]==le)
                                                                            & (regression_coeff_df["gate delay"]==gd)],
                                                                            axis=0)], columns=regression_coeff_df.columns)
                    median_regression_coeff_df = pd.concat([median_regression_coeff_df, append], ignore_index=True)
            self.regression_coeff_df = median_regression_coeff_df
        self.regression_settings = regression_settings
        #self.regression_polynom_str = self._get_baseline_regression_polynom_str()
        self.coeff_name_list = ["a%s"%(str(i).translate(SUB),) for i in range(0, len(self.regression_coeff_df.columns.to_list()[1:-3]))]
        self.model = self.Model(self)
    
    def summary_coeff_models(self, deg_max: int = 2, scoring: str = "fvalue"):
        summary_baseline_coeff_models = pd.DataFrame(data=None, columns=["model polynom", scoring])

        for idx in range(0,(self.regression_coeff_df.shape[1]-4)):
            coeff = Coefficient(self, idx, DEBUG=False)
            coeff.model.fit(deg_max=deg_max, scoring=scoring)
            summary_baseline_coeff_models = pd.concat([summary_baseline_coeff_models, pd.DataFrame({"model polynom": coeff.model.polynom_str, scoring: coeff.model.score}, index=[0])], ignore_index=True)
        fname = "summary_baseline_coeff_model"
        baseline_coeff_model_settings = vars(coeff.model)
        fpath = myutils.get_fpath(r"..\results", self.dataset_name, fname, "txt", baseline_coeff_model_settings,self.regression_settings)        
        if os.path.exists(fpath) == False:
            summary_baseline_coeff_models.to_csv(fpath)
        return summary_baseline_coeff_models, baseline_coeff_model_settings
    
    def model_assessment_plot(self, figsizex: float = 29.7, figsizey: float = 21, dpi: float=300, **kwargs):
        """ Generates three different plots to assess the model of each coefficient contained
        within the baseline polynom:
            - Prediction vs. Calculation
            - Residual plot
            - Histogram of residuals
        
        All three plots for eache coefficient are combined into one overview plot.
        
        Parameters
        ----------
        figsizex, figsizey : float, default=29.7 , 21
            figure size
        
        dpi : float, default=300
            resolution

        Keyword Arguments:
            Used to specifiy the modeling of coefficients
            * deg_max (int): specifies maximum degree of polynom including cross-terms (e.g. LE*GD)
            * scoring (str): scoring parameter used for coefficient modelling
            * transform_x, transform_y (bool): if True, x/y values will be transformed and scaled
        """
        
        fig, axes = plt.subplots(ncols=len(self.coeff_name_list), nrows=3,  figsize=(figsizex, figsizey), sharex = False, sharey = False)
        plt.subplots_adjust(wspace=0.3, hspace=0.2) # spacing between subplots
        fig.set_facecolor('white')
        fig.set_dpi(dpi) # adjust resolution
        fig.suptitle(self.dataset_name, x=0.5, y=0.98, ha='center', fontsize='xx-large', fontweight='bold')
        fig.text(x=0.5, y=0.95, s=self.model.overall_polynom_str, fontsize='xx-large', ha="center")
        for ax, col in zip(axes[0,:], self.coeff_name_list):
            ax.annotate(col, xy=(0.5,1.1), xytext=(0,30), xycoords='axes fraction',
                        textcoords='offset points', size = 'xx-large', weight='bold', ha='center', va='baseline')
        for ax, row in zip(axes[:,0], ["Prediction\nvs.\ncalculated", "Residual\nplot", "Residual\nhistogram"]):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-50,0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', weight='bold', ha='center', va='center', rotation='horizontal')
        for idx in range(len(self.coeff_name_list)):
            coeff = Coefficient(self, idx, DEBUG=False)
            coeff.model.fit(**kwargs)
            coeff.ypred = coeff.model.predict(coeff.X)
            subsubtitle = "%s \n%s: %s"%(coeff.model.polynom_str, coeff.model.scoring.upper(), "%.2f"%(coeff.model.score))
            axes[0,idx].set_title(subsubtitle, fontsize="small", pad=20, ha="center")
            coeff.model_assessment_plot("predvscalc", ax=axes[0,idx])
            coeff.model_assessment_plot("residualplot",ax=axes[1,idx])
            coeff.model_assessment_plot("residualhist",ax=axes[2,idx])
        fname = "assessment_baseline_coeff_model"
        baseline_coeff_model_settings = vars(coeff.model)
        fpath_fig = myutils.get_fpath(r"..\images", self.dataset_name,fname, "png", baseline_coeff_model_settings, self.regression_settings)
        fig.savefig(fpath_fig, dpi=dpi)
    
    def model_stabilty_plot(self, x: np.ndarray, figsizex: float = 12, figsizey: float = 9, dpi: int = 300):
        """ Plots the baseline model for 1000 randomized condition with the minimum/maximum
        boundaries of gate delay and laser energy
        
        Paramters
        ---------
        x : np.ndarray
            Array of wavelengths to be used for plotting
        
        figsizex, figsizey : float, default=29.7 , 21
            figure size
        
        dpi : float, default=300
            resolution 
        """
        le_random = np.random.uniform(np.min(self.le_list),np.max(self.le_list),[1000,])
        gd_random = np.random.uniform(np.min(self.gd_list),np.max(self.gd_list),[1000,])
        fig, ax = plt.subplots(figsize=(figsizex, figsizey), dpi=dpi, facecolor="white")
        fig.suptitle(self.dataset_name, fontweight="bold")
        fig.text(x=0.5, y=0.945, s="Predicted baselines of randomized conditions", fontweight="bold", ha="center")
        fig.text(x=0.5, y=0.925, s=self.model.overall_polynom_str, fontsize="small", ha="center")
        x_shift = mytrans.ShiftData(x_shift = self.x_shift).transform(x)
        for idx in range(1000):
            le = le_random[idx]
            gd = gd_random[idx]
            coeff_pred = self.model.predict(le, gd)
            poly_pred = np.poly1d(coeff_pred[::-1])
            ax.plot(x, poly_pred(x_shift), linewidth=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        fpath_fig = myutils.get_fpath(r"..\images", self.dataset_name, "baseline_prediction_stability", "png", self.model.baseline_coeff_model_settings, self.regression_settings)
        fig.savefig(fpath_fig, dpi=dpi)
            

    class Model:
        """ A class to represent the model of the baseline of a spectrum

        Parameters
        ----------
        baseline : object
            baseline class instance
        
        Attributes
        ----------
        baseline_coeff_model : list
            list containing Coefficient.Model object for each baseline coefficient

        polynom_str_list : list
            contains formatted polynom strings for each baseline coefficient
        
        overall_polynom_str : str
            formatted string of the fitted polynom
            e.g.: baseline model p(GD,LE,λ) = a₀(GD,LE) + a₁(GD,LE)·λ +  a₂(GD,LE)·λ²
        
        baseline_coeff_model_settings : dict
            settings applied for coefficient modelling
        """
        def __init__(self, baseline: object) -> None:
            self.baseline = baseline
        
        def fit(self, deg_max: int = 2, scoring: str = "fvalue", transform_x=True, transform_y=True, overwrite=False):
            """ Fits the model of all baseline coefficients of the baseline regression polynom with
            the coefficient obtained at baseline regression in dependency of laser energy and gate delay 
            and stores the model object at "..\models". To update the stored model later on, overwrite
            needs to be set to True.

            Parameters
            ----------
            deg_max : int, default=2
                specifies maximum degree of polynom including cross-terms (e.g. LE*GD)
            
            scoring : str, default="fvalue"
                scoring parameter used for coefficient modelling
            
            transform_x, transform_y : bool, default=True
                if True, x/y-values will be transformed and scaled prior modelling
            
            overwrite: bool, default=True
                overwrites the previously stored modelled
            """
            self.baseline_coeff_model = []
            self.polynom_str_list =[]
            self.coeff_list = []
            for idx in tqdm(range(len(self.baseline.coeff_name_list)), desc="Fit model"):
                coeff = Coefficient(self.baseline, idx, DEBUG=False)
                coeff.model.fit(deg_max, scoring, transform_x, transform_y)
                self.baseline_coeff_model.append(coeff.model)
                self.polynom_str_list.append(coeff.model.polynom_str)
                self.coeff_list.append(coeff.model.coeff_)
            fname = "baseline_model"
            self.baseline_coeff_model_settings = vars(coeff.model)
            fpath = myutils.get_fpath(r"..\models", self.baseline.dataset_name, fname, "pkl", self.baseline_coeff_model_settings, self.baseline.regression_settings)
            if (os.path.exists(fpath) == False) or overwrite==True:
                with open(fpath, "wb") as output_file:
                    pickle.dump(self, output_file)
            self.overall_polynom_str = self._get_overall_polynom_str()
            return self

        def _get_overall_polynom_str(self):
            """ Generates a formatted string of the polynom from baseline model
            
            Return
            ------
            polynom_str : str
                formatted polynom of baseline model
                e.g.: baseline model: p(GD,LE,λ) = a₀(GD,LE) + a₁(GD,LE)·λ +  a₂(GD,LE)·λ²
            """
            var_list = [i.replace("x", "\u03BB").replace("^","").translate(SUP) for i in  self.baseline.regression_coeff_df.columns.to_list()[1:-3]]
            polynom_str = "baseline model: p(GD,LE,\u03BB) = "
            for i in self.baseline.coeff_name_list:
                idx = self.baseline.coeff_name_list.index(i)
                if len(self.coeff_list[idx])==1 and self.coeff_list[idx][0] == 0: continue
                else:
                    if idx == 0:
                        polynom_str += "%s(GD,LE)"%i
                    else:
                        polynom_str += " + %s(GD,LE)\u00b7%s"%(i, var_list[idx])
            return polynom_str
        
        def predict(self, laser_energy: float, gate_delay: float):
            """ Predicts the baseline coefficients for given laser energy and gate delay

            Returns
            -------
            baseline_coeff_predict : list
                predicted coefficients
                e.g.: [a_0, a_1, a_2, a_3, a_4]            
            """
            baseline_coeff_predict = []
            X = pd.DataFrame({"LE": laser_energy, "GD": gate_delay}, index=[0])
            for model in self.baseline_coeff_model:
                coeff_predict = model.predict(X).values[0][0]
                baseline_coeff_predict.append(coeff_predict)
            return baseline_coeff_predict
    
class Coefficient:
    """ A class to represent a coefficient of the baseline of a dataset throughout modelation 
    
    The coefficients of a baseline need to modelled individually
    E.g.: the baseline polynom is given by p(gate delay, laser energy, lambda) = a_0(GD, LE) + a_1(GD, LE)*lambda + ... + a_i(GD, LE)*lambda^i
          a coefficient model is made up of a_i(GD, LE) = b0 + b1*GD + b2*LE + b3*GD^2 + b4*LE^2 + b5*LE*GD

    The subclass ``Coefficient.Model`` fits and predicts the coefficient models to return
    [b_0, b_1, ... , b_i] for a given gate delay and laser energy.     

    Parameters
    ----------
    baseline : object
        baseline class instance

    no : int
        number of coefficient
    
    DEBUG : bool, default=True
        if True, DEBUG message will be printed
    
    Attributes
    ----------
    name : str
        formatted name of the coefficient e.g. a₀
    
    X : pd.DataFrame
        dataframe containing laser energys and gate delays
    
    y : pd.DataFrame
        dataframe containing the coefficient
    
    ypred : pd.DataFrame
        predicted y-values

    model : object
        model of the coefficient
    """
    def __init__(self, baseline: object, no: int, DEBUG: bool = True) -> None:
        self.baseline = baseline
        self.no = no
        self.name = self.baseline.coeff_name_list[no]
        self.DEBUG = DEBUG
        if DEBUG == True:
            print("Coefficient: %s"%self.name)
        XY = self.baseline.regression_coeff_df
        X = XY.loc[:,["laser energy", "gate delay"]]
        self.X = X.rename(columns={"laser energy": "LE", "gate delay": "GD"})
        Y = XY.iloc[:, 1:-3]
        y = Y.iloc[:,self.no]
        self.y = y.to_frame(name=self.name)
        self.model = self.Model(self)

    def model_assessment_plot(self, type_: str, ax=None, figsizex: float = 9, figsizey: float = 9, dpi: int = 300):
        """ Generates a plot to assess the fit of the model for a coefficient

        Parameters
        ----------
        type_ : str
            specifies the type of the plot
            * "predvscalc" : predicted vs. calculated coefficients
            * "residualplot" : residual plot
            * "residualhist" : histogram of residuals
        
        ax : np.ndarray
            axes to be plotted
        
        figsizex, figsizey : float, default=9 , 9
            figure size
        
        dpi : float, default=300
            resolution
        """
        
        if ax==None:
            fig = plt.figure(figsize=(figsizex, figsizey), dpi=dpi, facecolor="white")
            fig.suptitle(self.baseline.dataset_name, fontweight = 'bold')
            subtitle = "Baseline Polynom: %s"%self.model.overall_polynom_str
            subsubtitle = "Modeled Coefficient: %s (%s: %s)"%(self.model.polynom_str, self.model.scoring.upper(), "%.2f"%(self.model.score))
            subsubsubtitle=""
            if type_=="residualplot": subsubsubtitle="Residual plot"
            if type_=="residualhist": subsubsubtitle="Histogram of residuals"
            if type_=="predvscalc": subsubsubtitle="Predicted vs. calculated"
            fig.text(x=0.5, y=0.945, s=subtitle, ha="center")
            fig.text(x=0.5,y=0.927,s=subsubtitle, fontsize="small", fontweight="bold", ha="center")
            fig.text(x=0.5, y=0.909, s=subsubsubtitle, fontsize="small", ha="center")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        residuals = self.y.to_numpy()-self.ypred.to_numpy()
        if type_=="residualplot":
            ax.scatter(self.ypred.index.to_list(),residuals)
            ax.axline((0,0),slope=0,color = "grey")
            ax.set_ylim(-np.max(np.absolute(residuals))*1.1,np.max(np.absolute(residuals))*1.1)
            ax.set_ylabel('Residuals')
        if type_ == "residualhist":
            ax.hist(residuals, density=True, bins="fd", ec="black")
            ax.set_xlabel("residuals")
        if type_=="predvscalc":
            ax.scatter(self.ypred, self.y)
            ax.axline((0,0),slope=1,color = "grey")
            ax.set_xlabel("Predicted coefficient (nm%s)"%str(self.no).translate(SUP))
            ax.set_ylabel("Calculated coefficient (nm%s)"%str(self.no).translate(SUP))

    class Model:
        """ A class to represent the model of a baseline coefficient.

        Parameters
        ----------
        coefficient : object
            coefficient class instance

        Attributes
        ----------
        deg_max : int
            maximum degree of fitted polynom including cross-terms (e.g. LE*GD)

        scoring : str
            scoring parameter used for coefficient modelling

        transform_x, transform_y : bool
            if True, x/y-values were transformed and scaled prior modelling

        pipeline : sklearn-pipeline
            pipeline used for fit and predict of the coefficients

        tfy : object
            Transformer for the y-values    
        """

        def __init__(self, coefficient: object):
            self.coefficient = coefficient
            self.median = self.coefficient.baseline.median

        def fit(self, deg_max: int = 2, scoring: str = "fvalue", transform_x: bool =True, transform_y: bool =True):
            """ Fits the coefficient model with values of baseline regression in dependency of laser energy
            and gate delay
            
            Parameters
            ----------
            deg_max : int, default=2
                specifies maximum degree of polynom including cross-terms (e.g. LE*GD)
            
            scoring : str, default="fvalue"
                scoring parameter used for coefficient modelling
            
            transform_x, transform_y : bool, default=True
                if True, x/y-values will be transformed and scaled prior modelling        
            """
            self.deg_max = deg_max
            self.scoring = scoring
            self.transform_x = transform_x
            self.transform_y = transform_y
            y =self.coefficient.y
            # PolynomialFeatures is separated within the baseline since the bias/intercept column should not be transformed and scaled
            self.pipeline = Pipeline([("polyfeatures",PolynomialFeatures(degree=self.deg_max, include_bias=False)),("addbias", PolynomialFeatures(degree=1)),("selector", mytrans.Selector(myreg.smaWLS(), method="bidirectional", scoring=self.scoring, DEBUG=False)), ("WLS", myreg.smaWLS(scoring=self.scoring))])
            if self.transform_x == True:
                self.pipeline.steps.insert(1, ("ptfx", PowerTransformer())) # inserts power transformer for x values at position 1 in pipeline
            if self.transform_y == True:
                self.tfy = mytrans.SkewScaleTransformer() # y values need to be transformed and scaled seperately since sklearn pipeline takes returns of transformer only for X values
                y = self.tfy.fit_transform(y)
            self.pipeline.fit(self.coefficient.X, y.values.reshape(-1,1)) # reshape necessary to match index with X values according to error msg
            self.feature_names_in_ = self.pipeline["selector"].best_coeff_names
            self.coeff_ = self.pipeline["selector"].best_coeff_values
            self.polynom_str = self._get_model_polynom_str()
            self.score = self.pipeline["selector"].best_score
            return self
        
        def predict(self, X: pd.DataFrame):
            """ Predicts the values of a coefficient in dependency of gate delay and laser energy

            Parameters
            ----------
            X : pd.DataFrame
                with following column structure:
                laser energy | gate delay
            
            Returns
            -------
            ypred : pd.DataFrame
                predicted coefficient(s)            
            """
            ypred = self.pipeline.predict(X)
            ypred = ypred.to_frame(name="%s"%self.coefficient.name)
            if self.transform_y == True:
                ypred = self.tfy.inverse_transform(ypred) # retransform coefficient
            self.coefficient.ypred = ypred
            return ypred
        
            # save pipeline
        def _get_model_polynom_str(self):
            """ Generates a formatted string of the coefficient model polynom

            Return
            ------
            polynom_str : str
                e.g. "a₀(GD,LE) = 1025.05 + 2.52e-1·LE + 3.00e0·GD²" 
            """
            coeff_formatted = ["%.2e"%i for i in self.coeff_]
            feature_names_formatted = [i.replace(" ", "\u00b7").replace("^","").translate(SUP) for i in self.feature_names_in_]
            polynom_str = "%s(GD,LE) = "%self.coefficient.name
            for i in coeff_formatted:
                idx = coeff_formatted.index(i)
                if self.coeff_[idx] != 0 or len(coeff_formatted)==1: # also write ai(LE,GD) = 0 eventhough coefficient is 0
                    if idx == 0:
                        polynom_str += "%s"%i
                    else:
                        polynom_str += " + %s\u00b7%s"%(i, feature_names_formatted[idx])
            return polynom_str
                    