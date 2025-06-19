# Overview of notebooks

These are Jupyter notebooks used to do the analyses and make figures for the manuscript.
The notebooks make use of the `scdynsys` package. The notebooks have to be executed in the right 
order to make sure that the right datasets have been created.

To get started with the CD8 data set, open the notebook `SequentialApproachClustering.ipynb` (for Leiden clustering),
and from there, either continue with `SequentialApproachFitting.ipynb` (for Stan model fitting), 
or with `IntegratedApproach.ipynb` (for VAE modeling).
Similar notebooks exist specific for the CD4 data.

## Description of the notebooks

### Analysis

* `SequentialApproachClustering.ipynb`: In this notebook we do the initial analyses for the SA. 
  We parse the flow data, cluster with the Leiden algorithm, find consensus clusters,
  and create timeseries.
* `SequentialApproachFitting.ipynb`: In this notebook we do the downstream analyses for the SA. 
  We fit Stan models to the timeseries and we also do model comparison.
* `IntegratedApproach.ipynb`: Fit the VAE model to sc-flow and cell count timeseries. Load pre-fitted models and 
  inspect the fit. Create posterior predictive checks. Save data for downstream plotting notebooks.
* `BiphasicModels.ipynb`: Fit simple biphasic models to the cell count data alone. Create a figure for the SI.
* `SequentialApproachClusCD4.ipynb`: Clustering notebook for the SA with the CD4 data.
* `SequentialApproachFitCD4.ipynb`: Fitting notebook for the SA with the CD4 data.
* `IntegratedApproachCD4.ipynb`: Fit the VAE model to CD4 sc-flow and cell count timeseries.
* `InfluxExperiment.ipynb`: Analyze data from a transfer experiment with congenic mice, 
  create an SI figure showing the results and a experimental diagram.
* `SensitivityAnalysis.ipynb`: Notebook for calculating sensitivity of the model w.r.t. the parameters,
  including the differentiation matrix elements $Q_{ij}$. 
* `ModelIdentifiability.ipynb`: Simulate data with the different models and fit all models to the 
  simulated datasets. Then do model comparison to see how well we can identify the *model* with a given dataset.
* `IdentifiabilityAnalysis.ipynb`: Practical identifiability analysis for the differentiation rates.
  Uses estimates for the SA.


### Figures

* `FigureDataOverview.ipynb`: Create the first figure of the manuscript that shows
  an overview of the data (cell counts, flow data, experimental setup).
  Also create SI figures for quality control, and anaylis of tetramer+ T cells.
* `FigureSAClustering.ipynb`: Here, we take the data processed in the SA clustering notebook
  and make a figure for the manuscript. Also creates a second version of the figure 
  that shows the fit of model IV to the data.
* `FigureSAFitting.ipynb`: Here, we take the data processed in the SA fitting notebook
  and make an SI figue for the manuscript showing fits to the 4 models.
* `FigureVAEDiagram.ipynb`: Make a diagram that explains how the IA model works.
* `FigureIAClusAndFit.ipnb`: Create the figure that shows the clustering and model fit for the integrated approach.
  This requires that we first run the `IntegratedApproach.ipynb` notebook.
* `FigureIAValidation.ipynb`: Create figure with additional posterior predictive checks specific for the IA.
  This requires that we first run the `IntegratedApproach.ipynb` notebook.
* `FigureSAClusteringCD4.ipynb`: Clustering figure in the sequential approach for the CD4 data (SI figure)
* `FigureSAFittingCD4.ipynb`: Model fitting figure in the sequential approach for the CD4 data (SI figure)
* `FigureIAClusFitValiCD4.ipynb`: Create IA figure for the CD4 lineage. Combines fitting, clustering and validation.
  This requires that we first run the `IntegratedApproachCD4.ipynb` notebook.
* `FigureIAValiCD4.ipynb`: Create additional validation figure for the IA and the CD4 lineage.
  This requires that we first run the `IntegratedApproachCD4.ipynb` notebook.
* `FigureBatchCorrect.ipynb`: Make plot that shows the results of the batch-correction analysis for a simple SI figure.
* `FigureClusCompare.ipynb`: Compare the clusters assigned by both methods using a heatmap of the Jaccard index (SI figure).
* `FigureIdentifiability.ipynb`: SI figures showing the results of the sensitivity and identifiability analyses.
  Both for CD8 and CD4 data. Requires running the `SensitivityAnalysis.ipynb`, `IdentifiabilityAnalysis.ipynb` and 
  `ModelIdentifiability.ipynb` notebooks.