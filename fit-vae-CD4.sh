#!/bin/bash

source venv/bin/activate

CLUS=12

fit_vae \
--seed 2344 \
--num_clus $CLUS \
--data_file data/pyro-dataset-CD4-conv-Lung-Aug23.csv \
--metadata_file data/pyro-dataset-meta-CD4-conv-Lung-Aug23.csv \
--countdata_file data/CD4_counts_Aug23.pkl \
--marker_file data/selected-markers-CD4-conv-Aug23.json \
--countdata_field "CD44+CD11a+_CD4_Tconv" \
--init_param_file data/init_param_guesses_CD4.json \
--init_time 8.0 \
--validation_index_file data/validation-idxs-Aug23-CD4-conv.txt \
--epochs 100000 \
--num_samples 100000 \
--output-file "CD4_modelIII_${CLUS}clus" \
--differentiation \
--distance_guided_diff \
--time_homogeneous


