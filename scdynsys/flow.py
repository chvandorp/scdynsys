import csv
import re
import numpy as np
from . import utilities as util
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Optional



mouse_name_pattern = re.compile("Mouse[_ ]?(\d+)")
day_pattern = re.compile(".*_d(\d+)_")
day_pattern2 = re.compile("^Day ?(\d+)_")
date_pattern = re.compile(".*_(\d{8})")


def extract_date(x):
    date = date_pattern.match(x).group(1)
    return date

def extract_mouseID(x):
    mouse_name = mouse_name_pattern.match(x).group(1)
    return mouse_name


def parse_sample_ID(x):
    mouse_name = extract_mouseID(x)
    day_num = day_pattern.match(x).group(1)
    date = extract_date(x)
    organ = util.find_organ_name(x)
    return f"Mouse{mouse_name}_{organ}_d{day_num}_{date}"

def import_flow_data(cell_type, exptID, organ, metadata_dict, verbose=False):
    """import .csv file content (parsed .fcs files)"""
    
    filename = f"../data/parsed_flow_data_{organ}_{cell_type}_{exptID}.csv" ## FIXME: this is terrible, Muriel
    with open(filename) as f:
        reader = csv.reader(f)
        table = [row for row in reader]
    header = table[0]
    table = table[1:]
    
    if verbose:
        for i, x in enumerate(header):
            print(i, x)
    
    sample_idx = header.index("sample")
    
    sample_IDs = [row[sample_idx] for row in table]
    mouse_IDs = [parse_sample_ID(x) for x in sample_IDs]

    included_mouse_IDs = [parse_sample_ID(x) for x in sorted(list(metadata_dict.keys()))]
    
    ## only keep those IDs that are in the metadata dict
    idxs = [i for i, ID in enumerate(mouse_IDs) if ID in included_mouse_IDs]
        
    sample_IDs = [sample_IDs[i] for i in idxs]
    table = [table[i] for i in idxs]
    
    all_markers = header[1:sample_idx]
    all_marker_values = np.array([row[1:sample_idx] for row in table], dtype=float)
    
    all_anns = header[sample_idx:]
    all_ann_values = np.array([row[sample_idx:] for row in table], dtype=str)
    
    unique_mouse_IDs = sorted(list(set(mouse_IDs)))

    day_nums = [metadata_dict[ID]["day"] for ID in mouse_IDs]
    unique_day_nums = sorted(list(set(day_nums)))
    
    num_samples_per_mouse = {
        ID : len([i for i in mouse_IDs if i == ID])
        for ID in unique_mouse_IDs
    }
    
    data_dict = [{
        "ID" : ID,
        "day" : day_nums[i],
        "expr" : dict(zip(all_markers, all_marker_values[i,:])),
        "ann" : dict(zip(all_anns, all_ann_values[i,:])),
    } for i, ID in enumerate(mouse_IDs)]
    
    return data_dict, num_samples_per_mouse, unique_mouse_IDs


def calc_frac(xs, gate):
    f_low = len([x for x in xs if x <= gate]) / len(xs)
    return np.array([f_low, 1-f_low])

def calc_nd_fracs(xss, gates):
    fracs = [calc_frac(xs, gate) for xs, gate in zip(xss, gates)]
    tp = fracs[0]
    for f in fracs[1:]:
        tp = np.tensordot(tp, f, axes=0)
    return tp



def remove_outliers(df: pd.DataFrame, sel_varnames: list, grouping: str=None, q=0.005):
    """
    remove outliers using the values of sel_varnames,
    restricting to grouping
    """
    if grouping is not None:
        groups = sorted(list(set(df[grouping].to_list())))
        print("groups: ", groups)
        cleaned_df = pd.DataFrame()
        for group in groups:
            sub_df = df[df[grouping] == group]
            cleaned_sub_df = remove_outliers(sub_df, sel_varnames, q=q)
            cleaned_df = pd.concat([cleaned_df, cleaned_sub_df], axis=0)
        return cleaned_df
    else:
        print("Number of events pre-cleanup: {}".format(len(df)))
        data_df = df[sel_varnames]
        no_outliers = (data_df < data_df.quantile(1-q)) & (data_df > data_df.quantile(q))

        cleaned_df = df[no_outliers.all(axis=1)].copy()
        print("Number of events post-cleanup: {}".format(len(cleaned_df)))

        ## remone NaNs
        cleaned_df.dropna(inplace=True, axis=0, how='any', subset=sel_varnames)
        print("Number of events post-cleanup: {}".format(len(cleaned_df)))
    
    ## give events new indices
    cleaned_df.reset_index(inplace=True, drop=True)
    
    return cleaned_df



def scale_to_unit_interval(
    df: pd.DataFrame, 
    sel_varnames: list, 
    grouping: str = None, 
    p: Optional[tuple[float, float]] = None
) -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=(-1,1))
    
    if grouping is not None:
        groups = sorted(list(set(df[grouping].to_list())))
        print("groups: ", groups)
        scaled_df = pd.DataFrame()
        for group in groups:
            sub_df = df[df[grouping] == group]
            scaled_sub_df = scale_to_unit_interval(sub_df, sel_varnames, p=p)
            scaled_df = pd.concat([scaled_df, scaled_sub_df], axis=0)
        return scaled_df
    else:
        print("Scaling data...")
        if p is None:
            scaled_array = scaler.fit_transform(df[sel_varnames])
        else:
            # apply a mask to values out of percentile range given by p
            pctls = np.percentile(df[sel_varnames], axis=0, q=p)        
            mask = (df[sel_varnames] < pctls[0]) | (df[sel_varnames] > pctls[1])
            masked_array = df[sel_varnames].copy()
            masked_array[mask] = np.nan
            scaler.fit(masked_array)
            scaled_array = scaler.transform(df[sel_varnames])
        scaled_df = pd.DataFrame(scaled_array, columns=sel_varnames)
        ## add metadata back
        for k in df.keys():
            if k not in sel_varnames:
                scaled_df[k] = df[k].values
    return scaled_df


def select_square_gate(
    xs: np.ndarray, 
    ys: np.ndarray, 
    xgate: tuple[float, float], 
    ygate: tuple[float, float]
) -> np.ndarray:
    """
    return a boolean array with True values whenever a point x, y is within the
    square gate xgate, ygate.

    Parameters
    ----------
    xs : np.ndarray
        x-coordinates of events.
    ys : np.ndarray
        y-coordinates of events.
    xgate : tuple[float, float]
        minimum and maximum x values.
    ygate : tuple[float, float]
        minimum and maximunm y values.

    Returns
    -------
    np.ndarray
        array with True and False values.

    """
    x1, x2 = xgate
    y1, y2 = ygate
    return (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
