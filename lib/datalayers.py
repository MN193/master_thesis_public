import numpy as np
from numpy import NaN
import pandas as pd
from matplotlib import pyplot as plt
import os
import pickle
import operator
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
import importlib

import lib.regressor as myreg
importlib.reload(myreg)

import lib.transformer as mytrans
importlib.reload(mytrans)

import lib.utils as myutils
importlib.reload(myutils)

class Dataset:
    """ A class to represent the dataset of a sample
    
    A dataset of a sample is made up of several txt-files for each measurement conditions varying laser energy and
    gate delay. Further, a condition is made up of 100 replicate spectra.

    The recommended file structure is:
    - ../Dataset as source directory
    - ../Dataset/Datasetname_E<laser energy>mJ_Gd<gate delay>us.txt for each condition
    - wavelengths (nm) | intensity 0 | ... | intensity 99 columnstructure for each condition file

    By initialisation, the data of each condition file is loaded and stored into a corresponding condition
    object within the DataFrame "fpath_le_cond_df". After processing each condition file, the dataset
    object is stored as a pickle file in "../temp". If a dataset instance is generated again
    for the same dataset, the dataset is retrieved from the correspoing pickle file. This cuts down
    computation time. The dataset pickle file is only recalculated and overwritten by setting the override parameter to True.

    Parameters
    ----------
    sdir : raw str
        source dirctory
    
    overwrite : bool, default = False
        recalculates and overwrite the dataset object
        if True: files contained in the dataset are loaded and stored into a pickle file
        if False: the dataset object is loaded if a pickle file already exists, otherwise it is
        loaded initially

    DEBUG : bool, default = True
        if True, debug messsages will be printed

    Attributes
    ----------
    name : str
        name of the dataset
    
    fpath_le_cond_df : pd.DataFrame
        DataFrame made up of filepath, laser energy, gate delay and condition instance with loaded
        data for each condition contained in the source directory

    wavelength_range : np.ndarray
        wavelengths of measurements after removal of dead pixels at the end of the spectra
    """
    @staticmethod
    def get_dataset_name(sdir: str):
            """ Extract the name of the dataset from source directory.
            
            Parameters
            ----------
            sdir : raw string
                    source directory (complete or relative)
            
            Return
            ------
            dataset_name : str
                        name of the dataset
            """
            idx_low = np.max([i for i, ltr in enumerate(sdir) if ltr == '\\'])
            dataset_name = sdir[idx_low+1:]
            dataset_name = dataset_name.replace("_","-")
            return dataset_name
    
    @staticmethod
    def get_fpath_le_gd_df(sdir: str, ftype: str = 'txt'):
        """ Stores all paths, laser energys and gate delays of the files contained in the source directory into
        an dataframe

        Parameters
        ----------
        sdir :  string or raw  string
                full or relative path of the source directory
        ftype : string 
                extension of the filetype (e.g.: 'txt')
        
        Return
        ------
        fpath_le_gd_df : pd.DataFrame
                DataFrame containing filepath (str) and corresponding laser energy (int), gate delay (int)
        
        """
        sdir = sdir.replace('/', '\\')
        ftype = '.%s'%(ftype)
        fpath_le_gd_df = pd.DataFrame(None, columns=["filepath", "laser energy", "gate delay","cond object"])
        for fname in os.listdir(sdir):
            if ftype in fname:
                fpath = os.path.join(sdir, fname)
                laser_energy, gate_delay = Condition.get_condition_parameters(fpath)
                new_row =  {"filepath": fpath, "laser energy": laser_energy, "gate delay": gate_delay}
                fpath_le_gd_df = pd.concat([fpath_le_gd_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        return fpath_le_gd_df.sort_values(by=["laser energy"], ignore_index=True)
    
    def __init__(self, sdir, overwrite: bool = False, DEBUG: bool = True) -> None:
        self.sdir = sdir
        self.name = Dataset.get_dataset_name(self.sdir)
        self.DEBUG = DEBUG
        self.fpath_le_gd_cond_df = Dataset.get_fpath_le_gd_df(self.sdir, ftype='txt')
        self._fpath_list = list(self.fpath_le_gd_cond_df["filepath"])
        self._laser_energy_list = list(np.sort(np.unique(self.fpath_le_gd_cond_df["laser energy"])))
        self._gate_delay_list = list(np.sort(np.unique(self.fpath_le_gd_cond_df["gate delay"])))
        if self.DEBUG==True: print("Dataset: %s"%self.name)
        self._read_in_all_conditions(overwrite=overwrite)
        self.wavelength_range = self._get_wavelength_range()

    def _read_in_all_conditions(self, overwrite: bool = False) -> None:
        """ Loads all files of a condition in a dataset and generates a corresponding condition
        object and stores it into the fpath_le_gd_cond attribute of the dataset. Afterwards, the
        dataset object is stored as a pickle objet.
        If the method is called again, the pickle object is retrieven instead of reloading all
        files. The dataset files are only reloaded if ``overwrite``=True.

        Parameters
        ----------
        overwrite : bool, default = False
            overwrites and stores the dataset object
            if True: files contained in the dataset are loaded and stored into a pickle file
            if False: the dataset object is loaded if a pickle file already exists, otherwise it is
            loaded initially
        """
        self.fpath_le_gd_cond_df["cond object"] = None
        fpath_obj = r"..\temp\%s_loaded_data.pkl"%(self.name)
        if (os.path.exists(fpath_obj) == False) or (overwrite == True):
            for fpath in tqdm(self._fpath_list, desc="Load dataset files"):
                idx = self.fpath_le_gd_cond_df.index[self.fpath_le_gd_cond_df["filepath"]==fpath].tolist()
                cond = Condition(dataset=self, fpath=fpath, DEBUG=False)
                self.fpath_le_gd_cond_df.loc[idx, "cond object"] = cond
            with open(fpath_obj, "wb") as output_file:
                pickle.dump(self.fpath_le_gd_cond_df, output_file)
        else:
            with open(fpath_obj, "rb") as input_file:
                self.fpath_le_gd_cond_df = pickle.load(input_file)
            print("Files of dataset loaded")
    
    def _get_wavelength_range(self) -> np.ndarray:
        """ Extracts the wavelengths of the measurements for the first condition and
        preprocesses them by interpolation and removal of dead pixels at the end of the spectra
        
        Returns
        -------
        x : np.ndarray
            wavelengths of measurements containend within dataset
        """
        cond = self.fpath_le_gd_cond_df.loc[[0], ["cond object"]].values[0][0]
        cond.interpolate_dead_pixels()
        return cond.x

    def summarize_baseline_coefficients(self, overwrite: bool = False, **kwargs) -> pd.DataFrame:
        """ Stores the polynomial coefficients and degree of the baseline for each replicate
        of all conditions into a dataframe.
        The dataframe is also savaed into "../results" as
        dataset_name_summary_baseline_coeff_baseline_regression_settings.txt

        If a summary with matching baseline settings for a dataset has already been stored and
        overwrite=False, the stored file is retrieven to cut computation time.

        Parameters
        ----------
        overwrite : bool, default=False
            If true, the dataframe is recalculated and not retrieven a previously stored file

        Keyword Arguments:
            Used to specifiy the baseline calculation
            * width_wvn (float): window width in nm
            * width_wvn_detail (float): window width with smaller steps at start/end of spectrum in nm
            * detailfactor (int): used to calculated the window steps within the detailed area by width_wvn/detail_factor
            * reverse (bool): if True, minimum filter runs in both directions
            * deg_min (int): minimum polynomial degree as constraint for baseline regression
            * deg_max (int): maximum polynomial degree as constraint for baseline regression             

        Returns
        -------
        summary_baseline_coeff : pd.DataFrame
            Dataframe with following column structure:
                x_shift | intercept | x | ... | x^deg_max | degree | laser energy | gate delay
        
        baseline_regression_settings : dict
            dictionary containing the baseline settings used throughout the calculation            
        """
        for fpath in tqdm(self.fpath_le_gd_cond_df["filepath"], desc="Calculate summary_baseline_coeff"):
            idx = self.fpath_le_gd_cond_df.index[self.fpath_le_gd_cond_df["filepath"]==fpath].tolist()
            cond = self.fpath_le_gd_cond_df.loc[idx,"cond object"].item()
            cond.interpolate_dead_pixels()
            cond_baseline_coeff_regression, baseline_regression_settings = cond.calculate_all_baselines(**kwargs)
            fpath_file = myutils.get_fpath(r"..\results", self.name, "summary_baseline_coeff_regression", "txt", baseline_regression_settings=baseline_regression_settings)
            if (os.path.exists(fpath_file) == False) or overwrite==True:
                if idx==[0]:
                    summary_baseline_coeff_regression = pd.DataFrame(None, columns=cond_baseline_coeff_regression.columns)
                summary_baseline_coeff_regression = pd.concat([summary_baseline_coeff_regression, cond_baseline_coeff_regression], ignore_index=True)
            else:
                summary_baseline_coeff_regression =  pd.read_csv(fpath_file, sep=",")
                return summary_baseline_coeff_regression, baseline_regression_settings
        summary_baseline_coeff_regression.to_csv(fpath_file, index=False)
        return summary_baseline_coeff_regression, baseline_regression_settings
    
    def _init_dataset_plot(self, subtitle: str, figsizex: float = 29.7, figsizey: float = 21, dpi: float=300, split: int = 1):
        """ Initializes figure and axes of of a plot to diplay a one or more plots for each condition.

        The grid of the plot is made up of gate delay (rows) x [laser energy x split] (columns).
        The dataset name is set as the main title.
        
        Parameters
        ----------
        subtitle : str
            detailed description of the plot
        
        figsizex, figsizey : float, default=29.7 , 21
            figure size
        
        dpi : float, default=300
            resolution

        split : int, default=1
            Value to define the amount of plots for each condition.
            If split > 1, the plots are displayed side by side.

        Returns
        -------
        fig : Figure
            Figure object

        axes : Axes
            Axes object

        y : float
            y-position of Figure subtitles to add extra subtitles later on 
        """
        # convert to string with units
        le_cols_str, gd_rows_str = [], []
        for i in self._laser_energy_list: le_cols_str.append('%d mJ'%(i))
        for i in self._gate_delay_list: gd_rows_str.append('%d µs'%(i))
        # set up figure, axes
        fig, axes = plt.subplots(ncols=len(le_cols_str)*split, nrows=len(gd_rows_str),  figsize=(figsizex, figsizey), sharex = False, sharey = False)
        plt.subplots_adjust(wspace=0.65, hspace=0.55) # spacing between subplots
        fig.set_facecolor('white')
        fig.set_dpi(dpi) # adjust resolution
        # label row and columns
        list = [axes[0, i] for i in range(0,(len(le_cols_str)*split),split)] # used to get the position of the label e.g. if split=2 only every second column is labeled due to two plots for each condition
        for ax, col in zip(list, le_cols_str):
            ax.annotate(col, xy=((0.5+(split-1)),1), xytext=(0, 30), xycoords='axes fraction',
                        textcoords='offset points', size = 'large', weight='bold', ha='center', va='baseline')
        for ax, row in zip(axes[:,0], gd_rows_str):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-50,0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', weight='bold', ha='center', va='center', rotation='horizontal')
        # label common axis
        fig.text(0.5, 0.92, 'Laserenergy', ha='center', fontsize='xx-large', fontweight='bold')
        fig.text(0.025,0.5,'Gatedelay', va='center', fontsize='xx-large', fontweight='bold', rotation='vertical')
        # set figure title
        figtitle = self.name
        y=1 if "simu" in subtitle else 0.97
        fig.suptitle(figtitle, x=0.5, y=y, ha='center', fontsize='xx-large', fontweight='bold')
        y -= 0.03
        fig.text(0.5, y, subtitle, ha='center', fontsize='xx-large')
        return fig, axes, y

    def plot_dataset(self, type_list: list[str], baseline_regression: bool = True, baseline_model: object = None, view: str = "detail", rep_no: int = 0, subtitle: str = None, figsizex: float = 29.7, figsizey: float = 21, dpi: float=300, DEBUG: bool = False, **kwargs):
        """ Plots one or more plot for each condition of a dataset in a grid

        The number of plots is defined by the list provided with type_list.

        If a model is given, the model is used for simulation and the simulation is
        combined with the given plot type.

        Parameters
        ----------
        type_list : list[str]
            List containing the types of the plot to be plotted
            If list contains more than one element, the plots will be plotted side by side.
            Available types:
        - "median": median spectra of each condition
        - "stability": median spectra of each condition with baseline of each replicate
        - "pie": distribution of polynomial degree throughout replicates
        - "single": spectrum of replicate no. ``rep_no``

        baseline_regression : bool = True
            If True, the baseline will be visible.
            For type="single", the pivot points will be visible as well
        
        baseline_model : object = None
            Object of fitted baseline model
        
        view : {"full", "detail"}, default="detail"
            Determines diyplay detail shown in the plots
            If view="detail", the baseline will be enlarged displayed.
        
        rep_no : str, default=0
            Number of replicate displayed if type="single"
        
        subtitle : str, default=None
            Subtitle of the plot
            If None, predefined subtitle for each type will be used. The default subtitles
            equal the description of each type within ``type_list``.

        figsizex, figsizey : float, default=29.7 , 21
            figure size
        
        dpi : float, default=300
            resolution
        
        Keyword Arguments:
            Used to specifiy the baseline calculation
            * width_wvn (float): window width in nm
            * width_wvn_detail (float): window width with smaller steps at start/end of spectrum in nm
            * detailfactor (int): used to calculated the window steps within the detailed area by width_wvn/detail_factor
            * reverse (bool): if True, minimum filter runs in both directions
            * deg_min (int): minimum polynomial degree as constraint for baseline regression
            * deg_max (int): maximum polynomial degree as constraint for baseline regression      
        """
        subtitle_dict = {"median": "median spectra of each condition",
                         "stability": "median spectra of each condition with baseline of each replicate",
                         "pie": "distribution of polynomial degree throughout replicates",
                         "single": "spectrum of replicate no. %d"%rep_no}
        
        fname_dict = {"median": "median_breg%s_bmod%s_v%s"%(baseline_regression, True if baseline_model!=None else False, view),
                      "stability": "stability_bmod%s_v%s"%(True if baseline_model!=None else False, view),
                      "pie": "pie",
                      "single": "single_%s_breg%s_v%s"%(rep_no, baseline_regression, view)}
        # modifiy subtitile with settings
        for type_ in type_list:
            if type_ == "median":
                conjunction_words = ["with", "and"]
                conj_count = 0
                if baseline_regression == True:
                    subtitle_dict[type_] += ' %s baseline regression'%(conjunction_words[conj_count])
                    conj_count += 1
                if baseline_model != None:
                    subtitle_dict[type_] += ' %s baseline simulation'%(conjunction_words[conj_count])
            if type_ == "stability":
                if baseline_model != None:
                    subtitle_dict[type_] += ' and simulated baseline'
        # generate subtitle        
        if subtitle == None:
            subtitle = ""
            for type_ in type_list:
                if len(type_list)>=2:
                    if 0 > type_list.index(type_) < (len(type_list)-1):
                        subtitle += ", "
                    if type_list.index(type_) == (len(type_list)-1):
                        subtitle += " and "
                subtitle += subtitle_dict[str(type_)]
        # default return arguments
        baseline_regression_settings = {}
        baseline_coeff_model_settings = {}
        # plot data
        fig, axes, y = self._init_dataset_plot(subtitle, figsizex=figsizex, figsizey=figsizey, dpi=dpi, split=len(type_list))     
        for fpath in tqdm(self.fpath_le_gd_cond_df["filepath"], desc="Calculate plot"):
            idx = self.fpath_le_gd_cond_df.index[self.fpath_le_gd_cond_df["filepath"]==fpath].tolist()
            cond = self.fpath_le_gd_cond_df.loc[idx,"cond object"].item()
            cond.DEBUG = DEBUG
            cond.interpolate_dead_pixels()
            le_0_list = []
            for i in self._laser_energy_list:
                le_0_list.extend([i, *[0]*(len(type_list)-1)])
            icol = le_0_list.index(cond.laser_energy)              
            irow = self._gate_delay_list.index(cond.gate_delay)
            pos_count = 0
            for type_ in type_list:
                if type_ in ["median", "stability", "pie"]:
                    baseline_regression_settings, baseline_coeff_model_settings, handles, labels = cond.plot_cond(ax=axes[irow][icol+pos_count], type_=type_, baseline_regression=baseline_regression, baseline_model=baseline_model, view=view, **kwargs)                                    
                    # only add subtitle for first iteration
                    if self.fpath_le_gd_cond_df["filepath"][self.fpath_le_gd_cond_df["filepath"]==fpath].index[0] == 0:
                        if len(labels)>=2:
                            fig.legend(handles, labels, loc="upper left", fontsize="x-large")
                        if baseline_model!=None:
                            y -= 0.02
                            fig.text(x=0.5, y=y, s=baseline_model.polynom_str_list[0], fontsize="large", ha="center")       
                if type_ == "single":
                    spectrum = Spectrum(dataset=self, cond=cond, repno = rep_no, DEBUG = DEBUG)
                    spectrum.baseline.find_pivot_points(**kwargs)
                    spectrum.baseline.calculate_regression(**kwargs)
                    spectrum.plot_spectrum(ax=axes[irow][icol+pos_count], baseline=baseline_regression, view=view)
                    baseline_regression_settings = vars(spectrum.baseline)
                pos_count += 1
        # build fpath for figure
        fname = "dataset_plot"
        for type_ in type_list: fname += "_" + fname_dict[type_]
        fpath_fig = myutils.get_fpath(r"..\images", self.name, fname, "png",  baseline_coeff_model_settings=baseline_coeff_model_settings, baseline_regression_settings=baseline_regression_settings)
        # save figure
        fig.savefig(fpath_fig, dpi = dpi)           

class Condition:
    """ A class to represent a condition of a dataset.

    A condition is made up of a txt file containing the measurement data of 100 replicate spectra.
    
    The recommended file structure is:
    - ../Dataset as source directory
    - ../Dataset/Datasetname_E<laser energy>mJ_Gd<gate delay>us.txt as filepath for each condition
    - wavelengths (nm) | intensity 0 | ... | intensity 99 as columnstructure for each condition file

    For intialisation, information to specifiy the dataset (``dataset`` or ``sdir``) and information to specify
    the condition (``fpath`` or ``laser_energy`` and ``gate_delay``) need to be passed.

    Parameters
    ----------
    dataset : object, default=None
        dataset class instance
        If None, a dataset object is generated corresponding to sdir
    
    sdir : raw str, default=None
        source dirctory of dataset
    
    fpath : raw str, default=None
        filepath of condition file
    
    laser_energy, gate_delay : int, default=None
        laser energy and gate delay of condition
    
    DEBUG : bool, default=True
        if True, debug messages will be printed
    
    Attributes
    ----------
    FDATA : np.ndarray
        data of condition txt file as numpy array
    x : np.ndarray
        x-values of measurements
    Y : np.ndarray
        y-values for all replicates
    """

    @staticmethod
    def get_condition_parameters(fpath: str):
        """Extract the laser energy and gate delay of a filepath
        
        Parameters
        ----------
        fpath : raw string
                file path (complete or relative)
        
        Return
        ------
        laser_energy, gate_delay : int
                                    laser energy or gate delay        
        """
        for unit in ['mJ', 'us']:
            idx_upper= fpath.index(unit)
            idx_low = 0
            i = idx_upper
            while idx_low == 0:
                i -= 1
                if fpath[i].isnumeric() == False:
                    idx_low = i+1
                    j = 0
                    while fpath[idx_low] == '0':
                        idx_low += 1
                        j += 1
                    result_unit = int(fpath[idx_low:idx_upper])
                    if unit == 'mJ': laser_energy = result_unit
                    if unit == 'us' : gate_delay = result_unit
        return laser_energy, gate_delay

    @staticmethod
    def get_median_intensity(Y : np.ndarray):
        '''
        Calculates the median intensity of all replicates

        Parameters
        ----------
        Y : np.ndarray
            y-values of replicates
        
        Returns
        -------
        ymed : np.ndarray
            rowwise median of Y
        '''
        ymed = np.median(Y, axis=1)
        return ymed
    
    def __init__(self, dataset=None, sdir: str = None, fpath: str = None, laser_energy: int = None, gate_delay: int = None, DEBUG: bool = True):
        if dataset!=None and sdir == None:
            self.dataset = dataset
        elif sdir!=None and dataset == None:
            self.dataset = Dataset(sdir)
        if (fpath==None and laser_energy!=None and gate_delay!=None):
            self.laser_energy = laser_energy
            self.gate_delay = gate_delay
            self.fpath = list(self.dataset.fpath_le_gd_cond_df[(self.dataset.fpath_le_gd_cond_df["laser energy"]==self.laser_energy) & (self.dataset.fpath_le_gd_cond_df["gate delay"]==self.gate_delay)]["filepath"])[0]
        elif (fpath!=None and laser_energy==None and gate_delay==None):
            self.fpath = fpath
            self.laser_energy, self.gate_delay = Condition.get_condition_parameters(self.fpath)
        if self.dataset.fpath_le_gd_cond_df[(self.dataset.fpath_le_gd_cond_df["filepath"]==self.fpath) ]["cond object"].tolist()[0] == None:
            # only used when creating a new dataset file
            self.FDATA = np.genfromtxt(self.fpath)
            self.x = self.FDATA[:, 0]
            self.Y = self.FDATA[:, 1:]
        else:
            # derieve condition parameters from (loaded) dataset pickle file
            self.dataset._fpath_list.index(self.fpath)
            self.__dict__ = self.dataset.fpath_le_gd_cond_df[(self.dataset.fpath_le_gd_cond_df["laser energy"]==self.laser_energy)&(self.dataset.fpath_le_gd_cond_df["gate delay"]==self.gate_delay)]["cond object"].tolist()[0].__dict__
        self.DEBUG = DEBUG
        if self.DEBUG == True: print("Condition: Laser energy: %d mJ | Gate delay: %d µs"%(self.laser_energy, self.gate_delay))

    def interpolate_dead_pixels(self, range_wvn: float = 2):
        """ Interpolates dead pixels with values  of a given range before and after the dead pixel and
        deletes dead pixxels at the end of the spectrum.

        Parameters
        --------
        range_wvn : float
                    range in nm to specify the values used to interpolate before and after the dead pixels
        """
        range = myutils.conv_wvntoidx(self.x, range_wvn)
        iszero = np.all(np.isin(self.Y, 0), axis=1) # checks with pixels of the wavelengths are dead for all replicates
        i = 0
        while i < len(self.Y[:,]):
            if iszero[i] == True: # checks if row is a dead pixel
                ifirst = i
                ilast = ifirst + 1
                while ilast < len(self.Y[:,]) and iszero[ilast] == True: # shifts ilast to the last row with deadpixels (exactly ilast+1 because in the intervall [ifirst:ilast] ilast is not included)
                    ilast += 1
                    i = ilast-1
                if ilast >= len(self.Y[:,]): # if the last rows of the data are made up of deadpixels, the rows get deleted
                    if self.DEBUG == True:
                        print('Empty pixels at the end of spectra deleted from %s nm to %s  nm' % ("{:.2f}".format(self.x[ifirst]), "{:.2f}".format(self.x[ilast-1])))
                    #logging.info(('Empty pixels at the end of spectra deleted from %s nm to %s  nm' % ("{:.2f}".format(x[ifirst]), "{:.2f}".format(x[ilast-1]))))
                    self.x = self.x[:ifirst]
                    self.Y = self.Y[:ifirst, :]
                else:
                    if range > ifirst: range = ifirst # checks if range does not exceed the rownumbers, fix for the case when ilast hits a dead pixel still necessary
                    x_c = self.x[ifirst:ilast] # rows with dead pixels
                    xp = np.append(self.x[(ifirst-range):ifirst], self.x[ilast:(ilast+range)]) # rows before and after the dead pixels (specified by range) # can this slicing be done more elegant?
                    column = 0
                    while column < self.Y.shape[1]: # repeats the interpolation for all replicate measurements
                        yp = np.append(self.Y[(ifirst-range):ifirst, column], self.Y[ilast:(ilast+range), column]) # yvalues before and after the dead pixels (specified by range)
                        self.Y[ifirst:ilast, column] = np.interp(x_c, xp, yp)
                        column += 1
                    if self.DEBUG == True:
                        print('Dead pixels interpolated from %s nm to %s  nm' % ("{:.2f}".format(self.x[ifirst]), "{:.2f}".format(self.x[ilast-1])))
                    #logging.info(('Dead pixels interpolated from %s nm to %s  nm' % ("{:.2f}".format(x[ifirst]), "{:.2f}".format(x[ilast-1]))))
            i += 1
        self.FDATA = np.empty((self.Y.shape[0], (self.Y.shape[1]+1))) # +1 for the x columns
        self.FDATA[:, 0] = self.x
        self.FDATA[:, 1:] = self.Y
    
    def calculate_all_baselines(self, **kwargs) -> tuple[pd.DataFrame, dict]:
        """ Calculates the baseline for each spectrum recoreded for the current condition
        and stores the coefficients into a DataFrame and the applied settings for baseline calculation
        into a dictionary
        
        Parameters
        ----------
        Keyword Arguments:
            Used to specifiy the baseline calculation
            * width_wvn (float): window width in nm
            * width_wvn_detail (float): window width with smaller steps at start/end of spectrum in nm
            * detailfactor (int): used to calculated the window steps within the detailed area by width_wvn/detail_factor
            * reverse (bool): if True, minimum filter runs in both directions
            * deg_min (int): minimum polynomial degree as constraint for baseline regression
            * deg_max (int): maximum polynomial degree as constraint for baseline regression

        Returns
        -------
        cond_baseline_coeff_regression : pd.DataFrame
            dataframe with following column structure
            x_shift | intercept | x | ... | x^deg_max | degree | laser energy | gate delay

        cond_baseline_regression_settings : dict
            dictionary containing the baseline settings used throughout the calculation   
        """
        n_replicates = self.FDATA.shape[1]-1 #-1 because wavenlengths are stored in first column
        
        for n in tqdm(range(n_replicates), desc="LE=%d mJ | GD=%d µs | Calculate baseline for each spectrum"%(self.laser_energy, self.gate_delay), disable=operator.not_(self.DEBUG)):
            spectrum = Spectrum(dataset = self.dataset, cond = self, repno=n, DEBUG = False)
            spectrum.baseline.find_pivot_points(**kwargs)
            spectrum.baseline.calculate_regression(**kwargs)
            col_names = ["x_shift"]
            col_names.extend(list(spectrum.baseline.deg_max_coeff_names))
            col_names.append("degree")
            if n==0:
                cond_baseline_coeff_regression = pd.DataFrame(None, columns=col_names)
            data = [spectrum.baseline.x_shift]
            data.extend(spectrum.baseline.coeff_values)
            for i in  range(((len(spectrum.baseline.deg_max_coeff_names)+1)-len(data))):
                data.append(0)
            data.append(spectrum.baseline.degree)
            data = dict(zip(col_names, data))
            cond_baseline_coeff_regression = pd.concat([cond_baseline_coeff_regression, pd.DataFrame(data, index=[0])], ignore_index=True)
            cond_baseline_coeff_regression["laser energy"] = self.laser_energy
            cond_baseline_coeff_regression["gate delay"] = self.gate_delay
        cond_baseline_regression_settings = vars(spectrum.baseline)
        return cond_baseline_coeff_regression, cond_baseline_regression_settings

    def calculate_median_baseline(self, **kwargs):
        """ Calculates the median baseline of all spectrums considerung only spectrums which feature
        the median common degree.
        
        Parameters
        ----------
        Keyword Arguments:
            Used to specifiy the baseline calculation
            * width_wvn (float): window width in nm
            * width_wvn_detail (float): window width with smaller steps at start/end of spectrum in nm
            * detailfactor (int): used to calculated the window steps within the detailed area by width_wvn/detail_factor
            * reverse (bool): if True, minimum filter runs in both directions
            * deg_min (int): minimum polynomial degree as constraint for baseline regression
            * deg_max (int): maximum polynomial degree as constraint for baseline regression
        
        Returns
        -------
        cond_baseline_coeff_regression_median : pd.DataFrame
            dataframe with following column structure for the median values of each column
            x_shift | intercept | x | ... | x^deg_max | degree | laser energy | gate delay
        
        cond_baseline_regression_settings : dict
            dictionary containing the baseline settings used throughout the calculation
        """
        cond_baseline_coeff_regression, cond_baseline_regression_settings = self.calculate_all_baselines(**kwargs)
        cond_baseline_coeff_regression = cond_baseline_coeff_regression[np.all(cond_baseline_coeff_regression.iloc[:,1:-3]!=0, axis=1)]     # remove unfitted spectra
        cond_baseline_coeff_regression = cond_baseline_coeff_regression[cond_baseline_coeff_regression["degree"]==np.median(cond_baseline_coeff_regression["degree"])]  # filter by median degree
        cond_baseline_coeff_regression_median =  pd.DataFrame(data=[np.median(cond_baseline_coeff_regression, axis=0)], columns=cond_baseline_coeff_regression.columns) 
        return cond_baseline_coeff_regression_median, cond_baseline_regression_settings
    
    def plot_cond(self, type_: str = "median", baseline_regression: bool = True, baseline_model: object = None, view: str = "full", ax: np.ndarray = None, ylim_bottom: float = None, ylim_top: float = None, figsizex: float = 12, figsizey: float = 9, dpi: int = 300, **kwargs):
        """ Plots data of the condition specified by type_

        Depending on the settings for baseline_regression or baseline_simulation, the calculated
        and/or simulated baseline will be combined with the given type.

        Parameters
        ----------
        type_ : {"median", "stability", "pie"}, default="median"
            Specifies the plot:
        - "median": median spectra
        - "stability": median spectra with baseline of each replicate
        - "pie": distribution of polynomial degree throughout replicates

        baseline_regression : bool = True
            If True, the baseline will be visible.
            For type="single", the pivot points will be visible as well
        
        baseline_model : object = None
            Object of fitted baseline model
        
        view : {"full", "detail"}, default="full"
            Defines the zoom level of the plot
            NOTE only used for type_="stability" or "median"

        ax : np.ndarray
            axes to be plotted

        ylim_bottom, ylim_top : float
            limits of y-axes

        figsizex, figsizey : float
            figure size for x and y dimension

        dpi : int
            resolution
        
        Keyword Arguments:
            Used to specifiy the baseline calculation
            * width_wvn (float): window width in nm
            * width_wvn_detail (float): window width with smaller steps at start/end of spectrum in nm
            * detailfactor (int): used to calculated the window steps within the detailed area by width_wvn/detail_factor
            * reverse (bool): if True, minimum filter runs in both directions
            * deg_min (int): minimum polynomial degree as constraint for baseline regression
            * deg_max (int): maximum polynomial degree as constraint for baseline regression        
        
        Returns
        -------
        baseline_regression_setting : dict, default={}
            dictionary containing baseline settings from the regression
        
        baseline_coeff_model_settings : dict, default={}
            dictionary containing settings form the baseline model

        handles, labels : list, default=[]
            list of handles and labels to create a legend
        """
        subsubtitle_dict = {"median": "median spectra",
                            "stability": "median spectra with baseline of each replicate",
                            "pie": "distribution of polynomial degree throughout replicates"}
        if type_ == "median":
            conjunction_words = ["with", "and"]
            conj_count = 0
            if baseline_regression == True:
                subsubtitle_dict[type_] += ' %s baseline regression'%(conjunction_words[conj_count])
                conj_count += 1
            if baseline_model != None:
                subsubtitle_dict[type_] += ' %s baseline simulation'%(conjunction_words[conj_count])
        if type_ == "stability":
            if baseline_model != None:
                subsubtitle_dict[type_] += ' and simulated baseline'
        if ax==None:
            y=0.98 if baseline_model==None else 0.985
            fig = plt.figure(figsize=(figsizex, figsizey), dpi=dpi, facecolor="white")
            fig.suptitle(self.dataset.name, y=y, fontweight = 'bold')
            subtitle = 'Laserenergy = %d mJ   Gatedelay = %d µs' % (self.laser_energy, self.gate_delay)
            y -= 0.035
            fig.text(x=0.5, y=y, s=subtitle, fontweight = 'bold', ha="center")
            y -= 0.018
            fig.text(x=0.5,y=y,s=subsubtitle_dict[type_],ha="center")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])        
        baseline_regression_settings, baseline_coeff_model_settings = {}, {}
        handles, labels = [], []
        if type_ in ["median", "stability"]:
            ymed = self.get_median_intensity(self.Y)
            ax.plot(self.x, ymed, linewidth=0.5)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
            max_list = [] # used for detailed view
            min_list = []
            plot_count = 0
            if type_ == "median":
                if baseline_regression==True:
                    baseline_regression_coeff_median, baseline_regression_settings = self.calculate_median_baseline(**kwargs)
                    x_shift = mytrans.ShiftData(x_shift = baseline_regression_coeff_median["x_shift"].values[0]).fit_transform(self.x)
                    median_baseline_regression_coeff_values = baseline_regression_coeff_median.iloc[:,1:-3].values[0]
                    poly_median = np.poly1d(median_baseline_regression_coeff_values[::-1])
                    ax.plot(self.x, poly_median(x_shift), label="Calculation", linewidth=3.5, color="burlywood")
                    max_list.append(max(poly_median(x_shift)))
                    min_list.append(min(poly_median(x_shift)))
                    plot_count +=1                   
            if type_ == "stability":
                n_replicates = self.FDATA.shape[1]-1 #-1 because wavenlengths are stored in first column
                for n in tqdm(range(n_replicates), desc="LE=%d mJ | GD=%d µs | Calculate baseline for each spectrum"%(self.laser_energy, self.gate_delay), disable=operator.not_(self.DEBUG)):   
                    spectrum = Spectrum(dataset = self.dataset, cond = self, repno=n, DEBUG = False)
                    spectrum.baseline.find_pivot_points(**kwargs)
                    spectrum.baseline.calculate_regression(**kwargs)
                    polyreg = np.poly1d(spectrum.baseline.coeff_values[::-1])
                    x_shift = mytrans.ShiftData(x_shift = spectrum.baseline.x_shift).fit_transform(self.x)
                    max_list.append(max(polyreg(x_shift)))
                    min_list.append(min(polyreg(x_shift)))
                    color="burlywood" if baseline_model!=None else None
                    if n < n_replicates-1:
                        ax.plot(self.x,polyreg(x_shift), linewidth=0.55, color=color)
                    else:
                        ax.plot(self.x,polyreg(x_shift), label="Calculation", linewidth=0.55, color=color)
                baseline_regression_settings = vars(spectrum.baseline)
                plot_count += 1
            if baseline_model != None:
                baseline_coeff_model_settings = baseline_model.baseline_coeff_model_settings
                if len(baseline_regression_settings) > 0:
                    regression_settings_to_compare = ["deg_min", "deg_max", "scoring", "minima_width_wvn", "minima_width_wvn_detail", "detailfactor", "reverse"]
                    for key in regression_settings_to_compare:
                        if vars(baseline_model.baseline)["regression_settings"][key] != baseline_regression_settings[key]:
                            print("baseline regression settings differ from calculated and simulated")
                            return baseline_regression_settings
                baseline_model_coeff_values = baseline_model.predict(laser_energy=self.laser_energy, gate_delay=self.gate_delay)
                poly_model = np.poly1d(baseline_model_coeff_values[::-1])
                x_shift = mytrans.ShiftData(x_shift = vars(baseline_model.baseline)["regression_settings"]["x_shift"]).fit_transform(self.x)
                if type_=="stability":
                    ax.plot(self.x, poly_model(x_shift), linewidth=3.5, color="red", linestyle="--", label = "Prediction")
                else:
                    ax.plot(self.x, poly_model(x_shift), label="Prediction", linewidth=3.5, linestyle="--", color="red")
                max_list.append(max(poly_model(x_shift)))
                min_list.append(min(poly_model(x_shift)))
                if "fig" in locals():
                    y -= 0.019
                    text = baseline_model.polynom_str_list[0]
                    fig.text(x=0.5, y=y, s=text, fontsize="x-small", ha="center")
            if view == "detail" and len(max_list) > 0 and len(min_list)>0:
                y_lim_bottom = min(min_list)-2000
                if max(max_list) <= 1000: y_lim_top = max(max_list)+2000
                else: y_lim_top = max(max_list)*2
                ax.set_ylim(y_lim_bottom, y_lim_top)
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) >= 2 and "fig" in locals():
                fig.legend(handles, labels, loc="upper left")
        if type_ == "pie":
            df, baseline_regression_settings = self.calculate_all_baselines(**kwargs)
            color_scheme = ["orchid", "goldenrod", "mediumseagreen", "steelblue", "coral"]
            df_group = df.groupby("degree").size()
            colors = [color_scheme[number-1] for number in df_group.index.tolist()]
            x = df_group.tolist()
            labels = df_group.index.tolist()
            ax.pie(x, labels=labels, colors=colors, autopct="%1.1f%%")                        
        return baseline_regression_settings, baseline_coeff_model_settings, handles, labels

class Spectrum:
    """ A class to represent a spectrum of a dataset.
    
    A spectrum is definied by x- (wavelength, nm) and y-values (intensity, a.u.) for a single
    measurement. The data for a specturm is stored among all other replicate measurements in a file
    for a specific condition (laser energy, gate delay).

    The recommended file structure is:
    - ../Dataset as source directory
    - ../Dataset/Datasetname_E<laser energy>mJ_Gd<gate delay>us.txt as filepath for each condition
    - wavelengths (nm) | intensity 0 | ... | intensity 99 as columnstructure for each condition file

    For initialisation, information to specifiy the dataset, condition and replicate needs to be passed:
    - dataset: ``dataset`` or ``sdir``
    - condition: ``cond`` or ``fpath`` or (``laser_energy`` and ``gate_delay``)
    - specturm: ``repno``

    Parameters
    ----------
    dataset : type[Dataset], default=None
        dataset class instance
        If None, a dataset object is generated corresponding to sdir
    
    sdir : raw str, default=None
        source dirctory of dataset
    
    cond : type[Condition], defaul=None
        condition class instance
        If None, a condition object is generated corresponding to fpath or laser_energy and gate_delay
        
    fpath : raw str, default=None
        filepath of condition file
    
    laser_energy, gate_delay : int, default=None
        laser energy and gate delay of condition
    
    repno : int, defaul=0
        replicate number
    
    DEBUG : bool, default=True
        if True, debug messages will be printed

    Attributes
    ----------
    y : str
        y-values of a spectrum

    baseline : type[Baseline]
        baseline class instance
    """
    def __init__(self, dataset: type[Dataset] =None, sdir: str =None, cond: type[Condition] =None, fpath: str = None, laser_energy: int = None, gate_delay: int = None, repno: int = 0, DEBUG: bool =True) -> None:
        if dataset!=None:
            self.dataset = dataset
        elif sdir!=None:
            self.dataset = Dataset(sdir)
        if cond != None:
            self.cond = cond
        elif fpath!= None:
            self.cond = Condition(dataset=self.dataset, fpath=fpath, DEBUG=DEBUG)
        elif (laser_energy!=None and gate_delay!=None):
            self.cond = Condition(dataset=self.dataset, laser_energy=laser_energy, gate_delay=gate_delay, DEBUG=DEBUG)
        self.repno = repno
        self.DEBUG = DEBUG
        self.y = self.cond.Y[:, repno]
        self.baseline = self.Baseline(self)
        if self.DEBUG == True: print("Spectrum: replicate no. %d"%(self.repno))

    def plot_spectrum(self, baseline: bool = True, view : str = "full", ax: np.array = None, ylim_bottom: float = None, ylim_top: float = None, figsizex: float = 12, figsizey: float = 9, dpi: int = 300):
        """ Plots specturm of a specific replicate measurement

        Parameters
        ----------
        baseline : bool, default=True
            if True, baseline will be plotted
        
        view : {"full", "detail"}, default="full"
            Defines the zoom level of the plot
            NOTE only works if baseline = True

        ax : np.array, default=None
            axes to be plotted

        ylim_bottom, ylim_top : float, default= None
            limits of y-axes

        figsizex, figsizey : float, default=12,9
            figure size for x and y dimension

        dpi : int, default=150
            resolution        
        """
        if ax==None:
            fig = plt.figure(figsize=(figsizex, figsizey), dpi=dpi, facecolor="white")
            fig.suptitle(self.dataset.name, fontweight = 'bold')
            subtitle = 'Laserenergy = %d mJ   Gatedelay = %d µs' % (self.cond.laser_energy, self.cond.gate_delay)
            fig.text(x=0.5, y=0.945, s=subtitle, fontweight = 'bold', ha="center")
            fig.text(x=0.5,y=0.927,s=("Replicate No. %d"%self.repno),ha="center")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])        
        ax.plot(self.cond.x, self.y, linewidth = 0.5, zorder=0)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
        if baseline == True:
            ax.scatter(self.x_min, self.y_min, marker='.', c='red', linewidths=1.0, zorder=10)
            if hasattr(self.baseline, "coeff_values")==True:
                polyreg = np.poly1d(self.baseline.coeff_values[::-1])
                x_shift = mytrans.ShiftData(x_shift = self.baseline.x_shift).fit_transform(self.cond.x)
                ax.plot(self.cond.x,polyreg(x_shift))
                ax.set_title('Degree of polynom: %d   %s: %s'%(self.baseline.degree, self.baseline.scoring.upper(), "{:.2f}".format(self.baseline.score)), fontsize='small', ha="center")
            if view == "detail":
                if hasattr(self.baseline, "coeff_values")==True: val = polyreg(x_shift)
                else: val = self.x_min
                y_lim_bottom = min(val)-2000
                if max(val) <= 1000: y_lim_top = max(val)+2000
                else: y_lim_top = max(val)*2
                ax.set_ylim(bottom=y_lim_bottom, top=y_lim_top)
    
    def correct_baseline(self):

        pass

    class Baseline:
        """ A class to represent the baseline of a spectrum.
        
        Parameters
        ----------
        spectrum : object
            spectrum class instance
        
        Attributes
        ----------
        xy_min : np.ndarray
            array containing x- and y-values of pivot points (result of minima filter)

        x_min, y_min : np.ndarray
            array containing the x- or y-values of pivot points (result of minima filter)
        
        width_wvn : float
            window width of minima filter in nm

        width_wvn_detail : float
            spectral range of minima filter with smaller steps at start/end of spectrum in nm

        detailfactor : int
            used to calculated the window steps within the detailed area by width_wvn/detail_factor

        reverse : bool
            True: minimum filter runs in both directions
            False: minimum filter runs in ascending order of x-values
        
        deg_min : int
                minimum polynomial degree as constraint for baseline regression
            
        deg_max : int
            maximum polynomial degree as constraint for baseline regression
        
        deg_max_coeff_names : np.ndarray[str]
            array containing strings of coefficient factors until deg_max
            e.g.: array(['intercept', 'x', 'x^2', 'x^3', 'x^4', 'x^5']) for deg_max=5

        coeff_names : list
            list of coefficient factors for the output polynom
            e.g.: ['intercept', 'x', 'x^2', 'x^3']

        degree : int
            degree of output polynom

        coeff_values : list
            numeric values of coefficients matching to coeff_names

        score : float
            score of output polynom

        x_shift : float
            shift value used to center x-values        
        """
        def __init__(self, spectrum):
            self.spectrum = spectrum
        
        def find_pivot_points(self, width_wvn: float = 40, width_wvn_detail: float = 20, detailfactor: int = 10, reverse: bool = True, **kwargs): # **kwargs necessary for dict input in dataset class, since same dict is used for find_pivot points and calculate baseline
            """ Moving minima filter with smaller window width (width/detail_factor instead of width) for 
            first and last width_wvn_detail nm of the spectrum to find pivot points. Steps are always current window width/2.
            The y-values are represented by the median value calculated with 4 values before and after the identified minima.
            
            Parameters
            ----------
            width_wvn : float, default=40
                window width in nm

            width_wvn_detail : float, default=20
                window width with smaller steps at start/end of spectrum in nm

            detailfactor : int, default=10
                used to calculated the window steps within the detailed area by width_wvn/detail_factor

            reverse : bool, default=True
                if True, minimum filter runs in both directions
            """
            def moving_minima_filter(x: np.ndarray, y: np.ndarray, x_min: np.ndarray, y_min: np.ndarray, limits: tuple[int, int], c_width: int, width_med_idx: int):
                """ Moving minima filter with width/2 steps
                
                Parameters
                ----------
                x, y : np.ndarray
                    data for x and y

                x_min, y_min : np.ndarray
                    arrays to store identified minimas

                limits : tuple[int, int]
                    tuple made up of lower limit and upper limit in which the filter operates

                c_width : int
                    current window width

                width_med_idx : int
                    half of the window width for the median calculation of y-values                
                """
                ifirst = limits[0]
                ilast = ifirst + c_width + 1 # +1 because selection [ifirst:ilast] is without ilast
                while ifirst < limits[1]:
                    c_array = y[ifirst:ilast]
                    min_val = min(c_array)
                    min_idx = np.where(c_array == min_val)[0][0]
                    y_med = np.median(y[((min_idx+ifirst)-width_med_idx):((min_idx+ifirst)+(width_med_idx+1))]) # add ifirst to match the indexes of fdata_interpolated instead of the index within the window
                    x_min = np.append(x_min, x[(min_idx+ifirst)])
                    y_min = np.append(y_min, y_med)
                    ifirst += int(c_width/2)
                    ilast += int(c_width/2)
                return x_min, y_min
            
            x = self.spectrum.cond.x
            y = self.spectrum.y
            x_rev = np.flipud(x)
            y_rev = np.flipud(y)
            self.minima_width_wvn = width_wvn
            if width_wvn_detail == 0:
                self.minima_width_wvn_detail = width_wvn
            else:
                self.minima_width_wvn_detail = width_wvn_detail
            if detailfactor == 0 or width_wvn_detail == 0:
                self.detailfactor = 1
            else:
                self.detailfactor = detailfactor
            self.reverse = reverse
            width = myutils.conv_wvntoidx(self.spectrum.cond.x, width_wvn)
            width_detail = myutils.conv_wvntoidx(self.spectrum.cond.x, self.minima_width_wvn_detail)
            width_med_idx = 4 # width of the window for median calculation of y-values
            x_min, y_min = np.empty((0,)), np.empty((0,)) # array for minima values
            c_width_list = [round(width/self.detailfactor), width, round(width/self.detailfactor)] # list of window widths used for each section
            limits_list = [(width_med_idx, width_detail), (width_detail, (len(self.spectrum.y) - width_detail)), ((len(self.spectrum.y) - width_detail), len(self.spectrum.y))] # list of limits used for each section
            for i in range(len(c_width_list)):
                c_width = c_width_list[i]
                limits = limits_list[i]
                x_min, y_min = moving_minima_filter(x,y,x_min,y_min,limits,c_width,width_med_idx)
                if reverse == True:
                    x_min, y_min = moving_minima_filter(x_rev,y_rev,x_min,y_min,limits,c_width,width_med_idx) # lets filter run in reverse direction -> uses fliped data as input

            x_min, y_min = x_min.reshape((len(x_min),1)), y_min.reshape((len(y_min), 1))
            self.spectrum.xy_min = np.hstack((x_min, y_min))
            self.spectrum.xy_min = np.unique(self.spectrum.xy_min, axis=0) # remove duplicates
            self.spectrum.x_min, self.spectrum.y_min = self.spectrum.xy_min[:,0], self.spectrum.xy_min[:,1]


        def calculate_regression(self, deg_min: int = 0, deg_max: int = 5, endpoint_weight: float = 10000, scoring: str = "fvalue", **kwargs): # **kwargs necessary for dict input in dataset class, since same dict is used for find_pivot points and calculate baseline
            """ Calculates the baseline by hiearchical polynomial regression with a force
            through the first and last three endpoints of the spectra

            Parameters
            ----------
            deg_min : int, default=0
                minimum polynomial degree as constraint for baseline regression
            
            deg_max : int, default=5
                maximum polynomial degree as constraint for baseline regression
                NOTE deg_max should not be >= 6 to avoid rounding errors 
            
            endpoint_weight : float, default = 10000
                Weight for the first and last datapoints
                Endpoint_weight >= 10000 equals a force through the endpoints
            
            scoring : {"fvalue", "naic", "nbic","nrmse"}, default = "fvalue"
                A single str to evaluate the predictions on the data       
            """
            self.deg_min = deg_min
            self.deg_max = deg_max
            self.endpoint_weight = endpoint_weight
            self.scoring = scoring
            set_config(transform_output="pandas")
            pipeline = Pipeline([("shift_x",mytrans.ShiftData(x_init=self.spectrum.cond.x)),("poly_features",PolynomialFeatures(degree=deg_max)),("selector", mytrans.Selector(myreg.smaWLS(endpoint_weight=self.endpoint_weight),method="hierarchical",deg_min=self.deg_min, scoring=self.scoring, DEBUG = self.spectrum.DEBUG)),("regressor", myreg.smaWLS(endpoint_weight=self.endpoint_weight, scoring=self.scoring))])
            x_min_df = pd.DataFrame(self.spectrum.x_min, columns=["x"])
            y_min_df = pd.DataFrame(self.spectrum.y_min, columns=["y"])
            pipeline.fit(x_min_df, y_min_df)
            self.deg_max_coeff_names = pipeline["poly_features"].get_feature_names_out()
            self.deg_max_coeff_names[0] = "intercept"
            self.coeff_names = pipeline["selector"].best_coeff_names
            self.degree = len(self.coeff_names)-1
            self.coeff_values = pipeline["selector"].best_coeff_values
            self.score = pipeline["selector"].best_score
            self.x_shift = pipeline["shift_x"].x_shift
            if self.spectrum.DEBUG == True:
                pipeline["regressor"].summary()