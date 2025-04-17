"""
Fit VAE models to static and/or time-stamped Flow Cytometry data

Use this script with slurm to run the models multiple times
"""
import csv
import pickle
import json
import numpy as np
import argparse # command-line arguments
import time # for measuring performance
import collections # Counter

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide.initialization import init_to_mean, init_to_value

from .vae.gmmdyn import VAEgmmdyn
from .vae.dyn import DynamicModelDiff, DynamicModel
from .vae.utils import onehot_encoding, train_test_loop_full_dataset

# start timer

tic = time.perf_counter()

########### extract cl arguments ###########

parser = argparse.ArgumentParser(
    description='Fit VAE model to Flow Cytometry Data.'
)
parser.add_argument(
    "--seed", type=int, default=144169,
    help="""seed for random number generator"""
)
parser.add_argument(
    "--data_file", type=str, required=True,
    help="""name of the file containing marker-expression data"""
)
parser.add_argument(
    "--metadata_file", type=str, required=True,
    help="""name of the file containing meta-data per cell"""
)
parser.add_argument(
    "--validation_index_file", type=str, default="",
    help="""name of the file containing indices of events used for validation"""
)
parser.add_argument(
    "--test_train_index_file", type=str, default="",
    help="""name of the file containing indices of events used for testing and training"""
)
parser.add_argument(
    "--countdata_file", type=str, required=True,
    help="""name of the file containing total cell count data"""
)
parser.add_argument(
    "--countdata_field", type=str, required=True,
    help="""field of the counts in the countdata file. e.g. 'CD44hi_CD8'."""
)
parser.add_argument(
    "--marker_file", type=str, default="",
    help="""name of the json file containing names of selected markers"""
)
parser.add_argument(
    "--init_param_file", type=str, default="",
    help="""name of file with initial parameter guesses"""
)
parser.add_argument(
    "--init_time", type=float, default=0.0,
    help="""the initial time of the ODE model (t_0)"""
)
parser.add_argument(
    "--epochs", type=int, default=20000,
    help="""number of epochs for the model fit"""
)
parser.add_argument(
    "--num_samples", type=int, default=100000,
    help="""number of cell samples used (subsampling from the full data set)"""
)
parser.add_argument(
    "--output_file", type=str, default="",
    help="""string used to create names for output files"""
)
parser.add_argument(
    "--num_clus", type=int, default=6,
    help="""number of components of the GMM"""
)
parser.add_argument(
    "--time_homogeneous", action="store_true", default=False, 
    help="""use the time homogeneous model"""
)
parser.add_argument(
    "--differentiation", action="store_true", default=False, 
    help="""estimate differentiation matrix Q"""
)
parser.add_argument(
    "--distance_guided_diff", action="store_true", default=False,
    help="""use the distance-guided differentiation model"""
)

args = parser.parse_args()

pyro.set_rng_seed(args.seed)

data_file_name = args.data_file
metadata_file_name = args.metadata_file
countdata_file_name = args.countdata_file
selected_markers_file_name = args.marker_file
init_param_file_name = args.init_param_file
validation_index_file_name = args.validation_index_file
test_train_index_file_name = args.test_train_index_file

COUNTDATA_FIELD = args.countdata_field

pyro_store_filename = f"results/parameter_store_{args.output_file}.dat"
loss_trace_filename = f"results/loss_trace_{args.output_file}.pkl"
settings_filename = f"results/settings_{args.output_file}.json"

# Run options
LEARNING_RATE = 2e-3
PERS_SHRINK_RATE = 1e-3
USE_CUDA = True

NUM_EPOCHS = args.epochs
TEST_FREQUENCY = 50
NUM_SAMPLES = args.num_samples
NUM_TEST_SAMPLES = 10000

NUM_CLUSTERS = args.num_clus
HIDDEN_DIM = 20
LATENT_DIM = 6
## make sure we start at the right day
T0 = args.init_time

TIME_HOMOGENEOUS = args.time_homogeneous
DIFFERENTIATION = args.differentiation
DISTANCE_GUIDED_DIFF = args.distance_guided_diff

print("number of populations:", NUM_CLUSTERS)


################################################
########## load a prepared data set ############
################################################

print("importing data...")

with open(data_file_name) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    raw_data_ext = np.array([row for row in reader], dtype=np.float32)

print("header of data file:", header)
print("first rows of data file:", raw_data_ext[:2])

## import selected markers
if selected_markers_file_name:
    with open(selected_markers_file_name, 'r') as f:
        sel_markers = json.load(f)
else: # use all available markers
    sel_markers = header
    
print("selected markers: ", sel_markers)

sel_markers_ext = header ## all markers in the data file
raw_data = raw_data_ext[:, [header.index(m) for m in sel_markers]]

print("importing meta-data...")

with open(metadata_file_name, 'r') as f:
    raw_metadata = [x for x in csv.reader(f)]
    metadata_varnames = raw_metadata[0]
    raw_metadata = np.array(raw_metadata[1:])

print("metadata:", metadata_varnames)
    
assert raw_data.shape[0] == raw_metadata.shape[0]
    
## subsample for fast testing/development
print("shape of full dataset", raw_data.shape)

# optionally, import previously sampled validation indices to exclude
if validation_index_file_name:
    with open(validation_index_file_name, 'r') as f:
        idxs_validate = np.loadtxt(f, dtype=int)
else:
    idxs_validate = []

# optionally load pre-determined test and train samples    
if test_train_index_file_name:
    with open(test_train_index_file_name, 'r') as f:
        idxs_test_train = np.loadtxt(f, dtype=int)    
else:
    idxs_test_train = [
        i for i in range(raw_data.shape[0]) 
        if i not in idxs_validate
    ]
    ## SUBSAMPLING!!!
    idxs_test_train = np.random.choice(idxs_test_train, NUM_SAMPLES, replace=False)


# import previously sampled train/test indices

assert len(set(idxs_test_train).intersection(set(idxs_validate))) == 0, "validation and test/train set overlap"

raw_data = raw_data[idxs_test_train, :]
raw_metadata = raw_metadata[idxs_test_train, :]

num_samples, feature_dim = raw_data.shape
num_test = NUM_TEST_SAMPLES
num_train = num_samples - num_test

test_indices = np.random.choice(raw_data.shape[0], num_test, replace=False)
train_indices = np.array([i for i in range(raw_data.shape[0]) if i not in test_indices], dtype=int)


#########################################
############# parse metadata ############
#########################################

raw_batch = raw_metadata[:,metadata_varnames.index("batch")]
unique_batch = sorted(list(set(raw_batch)))
num_batch = len(unique_batch)

num_samples_per_batch = collections.Counter(raw_batch)
batch_ref = max(num_samples_per_batch, key=lambda x: num_samples_per_batch[x])

print("choosing reference batch ID:", batch_ref)

print("number of batches:", num_batch)

if "expt" in metadata_varnames:
    raw_expt = raw_metadata[:,metadata_varnames.index("expt")]
else:
    raw_expt = np.zeros(raw_metadata.shape[0], dtype=int)

unique_expt = sorted(list(set(raw_expt)))
num_expt = len(unique_expt)

num_samples_per_expt = collections.Counter(raw_expt)
expt_ref = max(num_samples_per_expt, key=lambda x: num_samples_per_expt[x])

print("choosing reference expt ID:", expt_ref)

print("number of experiments:", num_expt)


sample_day = np.array(raw_metadata[:,metadata_varnames.index("day")], dtype=np.float32)
unique_sample_day = sorted(list(set(sample_day)))
num_sample_day = len(unique_sample_day)

print("number of sample days:", num_sample_day)


######## one-hot encoding of experimental batch #######

batch_expt_onehot, batch_onehot, expt_onehot = onehot_encoding(
    unique_batch, raw_batch, unique_expt, raw_expt, 0.1,
    batch_ref=batch_ref, expt_ref=expt_ref
)

print("batch encoding example:\n", batch_expt_onehot[0])
    

########### import data for dynamical model ############


with open(countdata_file_name, 'rb') as f:
    cell_counts = pickle.load(f)

scaling = 1e6

raw_count_data_combined = [
    (t-T0, x / scaling, ID) 
    for t, x, incl, ID in zip(
        cell_counts["DPI"], 
        cell_counts[COUNTDATA_FIELD],
        cell_counts["include"],
        cell_counts["ID"]
    ) 
    if t >= T0 and incl
]

raw_count_data_combined.sort()

incl_mouse_IDs_counts = [ID for t, x, ID in raw_count_data_combined]

print("included mice for count data:", incl_mouse_IDs_counts)


## create tensors (no need for data loaders)
ydata_raw = np.array([y[1] for y in raw_count_data_combined], dtype=np.float32)
ydata_tensor = torch.tensor(ydata_raw, dtype=torch.float32)
ytime_raw = np.array([y[0] for y in raw_count_data_combined], dtype=np.float32)
ytime_tensor = torch.tensor(ytime_raw, dtype=torch.float32)

xtime_raw = sample_day-T0
xtime_tensor = torch.tensor(xtime_raw, dtype=torch.float32)

## get unique times to decrease the number of computations

utime_raw = np.unique(np.concatenate([xtime_raw, ytime_raw]))
utime_tensor = torch.tensor(utime_raw, dtype=torch.float32)
print("unique time points: ", utime_tensor)

## compute indices
xtime_index_raw = np.array([np.where(np.isclose(utime_raw, t))[0][0] for t in xtime_raw])
xtime_index_tensor = torch.tensor(xtime_index_raw)

assert all(xtime_tensor == utime_tensor[xtime_index_tensor]), "indexing of unique times fails"

ytime_index_raw = np.array([np.where(np.isclose(utime_raw, t))[0][0] for t in ytime_raw])
ytime_index_tensor = torch.tensor(ytime_index_raw)

assert all(ytime_tensor == utime_tensor[ytime_index_tensor]), "indexing of unique times fails"

## import initial parameter guesses

with open(init_param_file_name, 'r') as f:
    init_param_dict = json.load(f)



#############################################
############ fit dynamical model ############
#############################################


# clear param store
pyro.clear_param_store()

raw_train_data = (
    raw_data[train_indices],
    xtime_index_raw[train_indices],
    batch_expt_onehot[train_indices],
)

raw_test_data = (
    raw_data[test_indices],
    xtime_index_raw[test_indices],
    batch_expt_onehot[test_indices],
)


################ setup the VAE ###################


# convert initial guesses to tensor
init_param_dict = {k : torch.tensor(v) for k, v in init_param_dict.items()}
# compute logX0 based on the number of clusters
logY0 = init_param_dict["logY0"]
init_param_dict["logX0"] = logY0.expand(NUM_CLUSTERS) - np.log(NUM_CLUSTERS)

if USE_CUDA:
    init_param_dict = {k : v.cuda() for k, v in init_param_dict.items()}

init_fn = init_to_value(values=init_param_dict, fallback=init_to_mean)

# we have to use the ODE solver in the non-autonomous case with differentiation
NUM_SOLVE = (not TIME_HOMOGENEOUS) and DIFFERENTIATION

# add a penalty for a long-term positive growth rate 
GROWTH_RATE_PENALTY = 10.0 ## TODO: parameter for the script. Keep up-to-date with Stan model

if DIFFERENTIATION:
    dynmod = DynamicModelDiff(
        NUM_CLUSTERS, hom=TIME_HOMOGENEOUS, init_fn=init_fn, 
        init_scale=0.01, numeric_solver=NUM_SOLVE, growth_rate_penalty=GROWTH_RATE_PENALTY,
        use_cuda=USE_CUDA
    )
else:
    dynmod = DynamicModel(
        NUM_CLUSTERS, hom=TIME_HOMOGENEOUS, init_fn=init_fn, init_scale=0.01, 
        growth_rate_penalty=GROWTH_RATE_PENALTY,
        use_cuda=USE_CUDA
    )


vae = VAEgmmdyn(
    data_dim=feature_dim, 
    z_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM, 
    num_clus=NUM_CLUSTERS,
    num_batch=batch_expt_onehot.shape[1],
    dyn=dynmod,
    time_scale=0.01,
    reg_scale_batch=1.0,
    reg_scale=10.0,
    reg_norm="l2",
    fixed_scales=True,
    distance_guided_diff=DISTANCE_GUIDED_DIFF,
    use_cuda=USE_CUDA
)

# save exact settings
with open(settings_filename, 'w') as f:
    json.dump(vae.settings_dict(), f, sort_keys=True, indent=4)

# setup the optimizer
adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)

# setup the inference algorithm
loss_method = Trace_ELBO(num_particles=20, vectorize_particles=True)
svi = SVI(vae.model, vae.guide, optimizer, loss=loss_method)

# training loop

raw_addl_data = (ydata_raw, ytime_index_raw, utime_raw)
train_elbo, test_elbo = train_test_loop_full_dataset(
    svi, 
    raw_train_data, 
    raw_test_data, 
    NUM_EPOCHS, 
    TEST_FREQUENCY,
    raw_addl_data=raw_addl_data,
    use_cuda=USE_CUDA,
    show_progress=False
)

######################################################
############ save result and diagnostics #############
######################################################

loss_trace = {
    "test" : test_elbo,
    "train" : train_elbo
}

with open(loss_trace_filename, 'wb') as f:
    pickle.dump(loss_trace, f)
    
store = pyro.get_param_store()
store.save(pyro_store_filename)

print("finished fitting dynamical model")

toc = time.perf_counter()

print(f"loading data and fitting model took {toc-tic:0.1f} seconds")



def main():
    pass
