# Master thesis
The aim of the master thesis is the empircal simulation of LIBS spectra.

## Project structure
* ``data``: data of datasets
  * the single data files should be stored according to: data/raw txt data/datasetname/datasetname_EXmJ_GdXus.txt
* ``databases``: databases to identify spectral lines
* ``images``: image output
* ``lib``: library with custom build modules for performing the master thesis
  * ``datalayers.py``: module used to represent the layers of data (dataset, condition, spectrum)
  * ``model.py``: module used to generate models for simulation of components of a spectrum (baseline, baseline coefficient)
  * ``regressor.py``: custom build regressor for scikit-learn library
  * ``transformer.py``: custom build transformers for scikit-learn library
  * ``utils.py``: util script
* ``models``: models used for modeling
* ``notebooks``: Jupyter notebooks for different steps of the master thesis:
  * ``NIST-1411``: summary of soft borosilicate glass dataset
  * ``ERM-EB316``: summary of AlSi12 dataset
  * ``SUS-1R``: summary of low allowed steel dataset
  * ``journal``: description of applied principles, thoughts, conclusions
  * ``merge_database`` : merges the atomtrace and imagelab databases
* ``results``: csv files of the output
* ``temp``: temporary stored pickle objects to cut computation time
* ``test``: scripts/notebooks to test and implement new functions

## Install Dependencies
* Install pipenv: https://pipenv.pypa.io/en/latest/
* Install dependencies with pipenv install
* To install a new package pkg: pipenv install pkg
