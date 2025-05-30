# Overview of notebooks

These are Jupyter notebooks used to do the analyses and make figures for the manuscript.
The notebooks make use of the `scdynsys` package. The notebooks have to be executed in the right 
order to make sure that the right datasets have been created.

## Description of the notebooks

### Analysis

* `SequentialApproachClustering.ipynb`: In this notebook we do the initial analyses for the SA. 
  We parse the flow data, cluster with the Leiden algorithm, find consensus clusters,
  and create timeseries.
* `SequentialApproachFitting.ipynb`: In this notebook we do the downstream analyses for the SA. 
  We fit Stan models to the timeseries and we also do model comparison.
* `IntegratedApproach.ipynb`: Fit the VAE model to sc-flow and cell count timeseries. Load pre-fitted models and 
  inspect the fit. Create posterior predictive checks. Save data for downstream plotting notebooks.
* `InfluxExperiment.ipynb`: **IN PROGRESS**
* `BiphasicModels.ipynb`: Fit simple biphasic models to the cell count data alone. Create a figure for the SI.


### Figures

* `FigureDataOverview.ipynb`: Create the first figure of the manuscript that shows
  an overview of the data (cell counts, flow data, experimental setup).
* `FigureSAClustering.ipynb`: Here, we take the data processed in the SA clustering notebook
  and make a figue for the manuscript.
* `FigureVAEDiagram.ipynb`: Make a diagram that explains how the IA model works.
* `FigureIAClusAndFit.ipnb`: Create the figure that shows the clustering and model fit for the integrated approach.
  This requires that we first run the `IntegratedApproach.ipynb` notebook.
* `FigureIAValidation.ipynb`: Create figure with additional posterior predictive checks specific for the IA.
  This requires that we first run the `IntegratedApproach.ipynb` notebook.
* `FigureSAClusteringCD4.ipynb`: Clustering figure in the sequential approach for the CD4 data (SI figure) **IN PROGRESS**
* `FigureSAFittingCD4.ipynb`: Model fitting figure in the sequential approach for the CD4 data (SI figure) **IN PROGRESS**
