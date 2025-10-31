# Translating Markov Models from R to Python

The goal of this repository is to translate an existing implementation of Markov models in R to Python. The original R code can be found in the [`r_version/`](r_version/) directory, or in the [viewsforecasting repository](https://github.com/prio-data/viewsforecasting/tree/main/Tools/new_markov). 

## Structure of repository

- [`r_version/`](r_version/): Contains the original R implementation of the Markov model.
- [`notebooks/`](notebooks/): Contains Jupyter notebooks for experimentation and testing of Markov models.
    - [`sandbox.ipynb`](notebooks/sandbox.ipynb): A notebook for testing and experimenting with the Markov model implementation.
    - [`fetch_test_data.ipynb`](notebooks/fetch_test_data.ipynb): A notebook to fetch a test dataset from VIEWSER and save it locally.
- [`data/`](data/): Directory to store datasets
- [`src/`](src/): Contains the source code for the Markov model implementation in Python.
    - [`markov_model.py`](src/markov_model.py): The main module implementing the Markov model.
    - [`auxiliaries.py`](src/auxiliaries.py): Contains some misc helper functions (currently not used).