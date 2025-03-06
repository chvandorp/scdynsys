"""
Script for fitting Stan models to T-cell data
"""

import argparse # command-line arguments
import pickle
import arviz
import os

from .stanfit import fit_stan_model


############################################
####### Parse command-line arguments #######
############################################


parser = argparse.ArgumentParser(
    description='Fit Stan Model to Clustered Flow Cytometry Data.'
)
parser.add_argument(
    "--data_file", dest="data_file", type=str,
    help="""name of the data set. This must be a pickle (.pkl)"""
)
parser.add_argument(
    "--model_spec_file", dest="model_spec_file", type=str,
    help="""name of the file with model specifications. This must be a pickle (.pkl)"""
)
parser.add_argument(
    "--output_dir", dest="output_dir", type=str,
    help="""name of the directory to save output"""
)
parser.add_argument(
    "--stan_output_dir", dest="stan_output_dir", type=str,
    help="""name of the directory to save raw output produced by Stan"""
)


args = parser.parse_args()

with open(args.data_file, 'rb') as f:
    dataset = pickle.load(f)
    
with open(args.model_spec_file, 'rb') as f:
    model_specs = pickle.load(f)

def file_core(fn):
    return os.path.splitext(os.path.basename(fn))[0]

ds_core = file_core(args.data_file)
mn_core = file_core(args.model_spec_file)


############################################
########### Fit the stan model #############
############################################

print(f"starting stan model fit for data set {args.data_file} and model {args.model_spec_file}")

stan_output_dir = os.path.join(args.stan_output_dir, f"{ds_core}_{mn_core}")

sam = fit_stan_model(
    dataset["count_times"],
    dataset["count_data"],
    dataset["freq_times"], 
    dataset["freq_data"],
    dataset["t0"],
    model_specs["obs_model_freq"], 
    model_specs["time_dependence"], 
    model_specs["population_structure"],
    count_scaling=1e6,
    iter_warmup=1000,
    iter_sampling=1000,
    thin=4,
    chains=4,
    show_progress=False,
    show_console=True,
    output_dir=stan_output_dir
)


asam = arviz.from_cmdstanpy(sam, log_likelihood="log_lik")
loo  = arviz.loo(asam, pointwise=True)


############################################
############ save the results ##############
############################################

result = {
    "asam" : asam,
    "loo" : loo,
    "stan_vars" : sam.stan_variables()
}

output_file = os.path.join(args.output_dir, f"samples_{ds_core}_{mn_core}.pkl")

with open(output_file, 'wb') as f:
    pickle.dump(result, f)
    
    

def main():
    pass



