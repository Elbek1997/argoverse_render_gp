from utils import *

import pickle as pkl
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import threading
from typing import List

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import lane_waypt_to_query_dist, get_nt_distance, get_oracle_from_candidate_centerlines, \
    get_normal_and_tangential_distance_point, remove_overlapping_lane_seq, filter_candidate_centerlines, get_centerlines_most_aligned_with_trajectory
from argoverse.utils.mpl_plotting_utils import plot_lane_segment_patch, visualize_centerline
from shapely.geometry import LinearRing, LineString, Point, Polygon

import pandas as pd
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from math import ceil

import pdb

from copy import copy

root_dir = "dataset/test"
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
loader = ArgoverseForecastingLoader(root_dir)


# Output
output_file = "output/v_4/test.pkl"
 
print("[INFO]Loading Map...")
avm = ArgoverseMap()



print("[INFO]Loading Dataset %s"%root_dir)

length = len(loader)


recs = []
for i in tqdm(range(length), desc="Loading Argoverse"):
    recs.append( copy(loader[i]) )

print('Total number of sequences:', length)


def thread_run(recs, label):


    print("Starting %s"%label)
    ret = []

    for i in tqdm(range(len(recs)), desc="Processing %s"%label):
        
        rec = recs[i]
        
        # City
        city = rec.city

        # ID
        id = int(rec.current_seq.name[:-4])
        

        # Source
        traj = rec.agent_traj
        source = traj[:20]
        # target = traj[20:]
        
        # Candidates
        candidate_cl, _ = get_candidate_centerlines(avm, source, city)


        if len(candidate_cl)==0:
            print("[INFO] Zero candidates")

        # # Ground truth
        # label = closest_line(candidate_cl, target)

        
        ret.append( { "id": id, "city" : city, "source_trj": source, "candidate_cl": candidate_cl
                    # , "target_trj": target, "label": label 
                    } )
                
    
    return ret


# thread_run(recs, "output/multi_candidate/v2/train.pkl")

n_jobs = 10
batch_size = ceil(length/n_jobs)

ret = Parallel(n_jobs=n_jobs)(delayed(thread_run)(recs[i*batch_size: (i+1)*batch_size], "%d"%(i+1)) for i in range(ceil(length/batch_size) ))

agg = []

for data in ret:
    agg = agg + data


save_pickle(agg, output_file)

print("Done")

# threads = threading.enumerate()

# print("Waiting background jobs to finish...")
# for thread in threads:

#     if thread!=threading.currentThread():
#         thread.join()

    