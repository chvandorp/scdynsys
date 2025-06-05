# scdynsys
Methods for fitting dynamical models to single-cell data

Our preprint about this method and application to resident-memory T cell data
can be found on BioRxiv:

> CH van Dorp, JI Gray, DH Paik, DL Farber, AJ Yates. 
> A Variational deep-learning approach to modeling memory T cell dynamics 
> [preprint](https://www.biorxiv.org/content/10.1101/2024.07.08.602409v1) (2024)

> [!NOTE]
> This repository is under construction and will be regularly updated.


## Installation and testing

To install the `scdynsys` package, first clone the git repository,
and then create a virtual environment

```bash
git clone git@github.com:chvandorp/scdynsys.git
cd scdynsys
python3 -m venv .venv
```

Here it is assumed that python3 points to a python version >= 3.10.
Next, activate the virtual environment and install the package.
This will download and install all dependencies.

```bash
source .venv/bin/activate
pip install .
```

You can use `pytest` to see if something isn't working.
This will collect all unittests in the `tests` folder and run them.

```bash
pip install pytest
python3 -m pytest
```

## Using the package

The `notebooks` folder contains code used for the aforementioned preprint.
To run the notebooks, you first have to download the flow cytometry data from Zonedo *TODO: link to data repo: create DOI*.
The `notebooks` folder contains an overview (README) file describing what all notebooks are used for and the order in which
they have to be executed.


## TODO

Some improvements:

* Replace python `pickle`s in the data folder with human-readable `json` files. 
* Refactor the notebooks to avoid code repetition between the CD8 and CD8 analysis.
* Script to automatically download the data from Zonedo.