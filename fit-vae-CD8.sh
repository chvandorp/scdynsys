#!/bin/bash

source venv/bin/activate

CLUS=10

fit_vae \
--seed 1202 \
--num_clus $CLUS \
--data_file data/pyro-dataset-CD8-Lung-Aug23.csv \
--metadata_file data/pyro-dataset-meta-CD8-Lung-Aug23.csv \
--countdata_file data/CD8_counts_Aug23.pkl \
--marker_file data/selected-markers-CD8-Aug23.json \
--countdata_field "CD44+CD11a+_CD8" \
--init_param_file data/init_param_guesses_CD8.json \
--init_time 8.0 \
--validation_index_file data/validation-idxs-Aug23.txt \
--epochs 100000 \
--num_samples 100000 \
--output_file "CD8_modelIV_${CLUS}clus" \
--differentiation \
--distance_guided_diff \
--no-time_homogeneous
