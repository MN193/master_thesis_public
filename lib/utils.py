import numpy as np

def conv_wvntoidx(x: np.ndarray, wvn: float):
    """ Converts the desired wavenumber range into row numbers/indexes of a given spectrum.

    Parameters
    ----------
    x : np.array
            x-values of a spectra
    wvn : float
            wavenumber in nm

    Return
    -------
    range : int
            number of rows for the specified wavenumber              
    """
    diff = abs(x[1] - x[0])
    range = round(wvn/diff)
    return range

def get_fpath(directory: str, datasetname: str, name: str, file_ext: str, baseline_coeff_model_settings: dict = {}, baseline_regression_settings: dict = {}):
    """ Generates filepath for a new file

    Structure of fpath:
    directory/datasetname/datasetname_name_baseline_settings.file_ext
    
    Parameters
    ----------
    directory: str
        direcotry of file e.g.: r"..\results"
    
    datasetname : str
        name of dataset

    name : str
        specific name to specifiy file e.g. "summary_baseline_coefficients"
    
    baseline_coeff_model_settings : dict
        dictionary of settings applied for baseline coefficient modeling
    
    baseline_regression_settings : dict
        dictionary of settings applied for baseline regression
    
    Returns
    -------
    fpath : str
        file path generated from the given parameters
    """
    # transform baseline_coeff_model_settings to str
    if len(baseline_coeff_model_settings) < 1:
        baseline_coeff_model_settings_str = ""
    else:
        baseline_coeff_model_settings_str = "_bcmod"
        baseline_coeff_model_settings_names = {"deg_max": "dmax", "scoring": "", "median": "med","transform_x": "trfx", "transform_y": "trfy"}
        for key in baseline_coeff_model_settings_names.keys():
            baseline_coeff_model_settings_str += "_%s%s"%(baseline_coeff_model_settings_names[key], baseline_coeff_model_settings[key])
    # transform baseline_regrssion_settings to str
    if len(baseline_regression_settings) < 1:
        baseline_regression_settings_str = ""
    else:
        baseline_regression_settings_str = "_breg"
        baseline_regression_settings_names = {"deg_min": "dmin", "deg_max": "dmax","scoring": "", "endpoint_weight": "endw","minima_width_wvn": "minwidth", "minima_width_wvn_detail": "minwidthdet", "detailfactor": "detf" ,"reverse": "rev"}
        for key in baseline_regression_settings_names.keys():
            baseline_regression_settings_str += "_%s%s"%(baseline_regression_settings_names[key], baseline_regression_settings[key])
    file_ext = file_ext.replace(".","")
    fpath = r"%s\%s\%s_%s%s%s.%s"%(directory, datasetname, datasetname, name, baseline_coeff_model_settings_str, baseline_regression_settings_str, file_ext)
    return fpath
        