# Overview of notebooks

These are Jupyter notebooks used to do the analyses and make figures for the manuscript.
The notebooks make use of the `scdynsys` package. The notebooks have to be executed in the right 
order to make sure that the right datasets have been created.

## Description of the notebooks

* `FigureDataOverview.ipynb`: Create the first figure of the manuscript that shows
  an overview of the data (cell counts, flow data, experimental setup).
* FigureVAEDiagram.ipynb`: Make a diagram that explains how the IA model works.
* `SequentialApproachClustering.ipynb`: In this notebook we do the initial analyses for the SA. 
  We parse the flow data, cluster with the Leiden algorithm, find consensus clusters,
  and create timeseries. **IN PROGRESS**
* `SequentialApproachClustering.ipynb`: In this notebook we do the initial analyses for the SA. 
  We parse the flow data, cluster with the Leiden algorithm, find consensus clusters,
  and create timeseries. **IN PROGRESS**
* `SequentialApproachFitting.ipynb`: In this notebook we do the downstream analyses for the SA. 
  We fit Stan models to the timeseries and we also do model comparison. **IN PROGRESS**
* `FigureSAClustering.ipynb`: Here, we take the data processed in the SA clustering notebook
  and make a figue for the manuscript.
* `IntegratedApproach.ipynb`: **IN PROGRESS**
* `InfluxExperiment.ipynb`: **IN PROGRESS**
* `BiphasicModels.ipynb`: Fit simple biphasic models to the cell count data alone.
Create a figure for the SI.
* **TODO**
