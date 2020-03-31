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

from copy import deepcopy as dc

# Input
dataset = "input/val_raw.pkl"
# Output
output_file = "output/v5/val.pkl"
 
print("[INFO]Loading Map...")
avm = ArgoverseMap()

city_lane_centerlines_dict = dc(avm.city_lane_centerlines_dict)


print("[INFO]Loading Dataset %s"%dataset)

limit = None
recs = pkl.load(open(dataset, "rb"))[:limit]
length = len(recs)

print('Total number of sequences:', length)


def thread_run(recs, label):

    #region Borrow Limit
    borrow_limit = 5
    #endregion
    print("Starting %s"%label)
    ret = []

    for i in tqdm(range(len(recs)), desc="Processing %s"%label):
        
        rec = recs[i]
        
        # City
        city = rec["city"]

        # ID to lane dict
        lane_centerlines_dict = avm.city_lane_centerlines_dict[city]
        lane_polygons_dict = avm.city_to_lane_polygons_dict[city]

        # Source and target
        source = rec["traj"][:20].tolist()
        target = rec["traj"][20:].tolist()
        
        #region Closest centerlines

        #region Target
        target_ids = find_closest_centerline_v2(avm, target, city, additional=source[-borrow_limit:], from_back=False)
        # Angle between centerline and trajectory
        angle = angle_between_lines( crop_line(build_line(target_ids, lane_centerlines_dict), target[0], target[-1]), target )
        target_closest_line = lane_centerlines_dict[ target_ids[ -1 if angle>=90 else 0 ] ]
        #endregion

        #region Source
        source_ids = find_closest_centerline_v2(avm, source, city, additional=target[:borrow_limit+1], from_back=True)
        # Angle between centerline and trajectory
        angle = angle_between_lines( crop_line(build_line(source_ids, lane_centerlines_dict), source[0], source[-1]), source )
        # Select line before last one from correct side
        index = (1 if angle>=90 else -2) if len(source_ids) > 1 else 0
        source_closest_line = lane_centerlines_dict[ source_ids[index] ]

        # Headstart
        headstart = crop_line(build_line([source_closest_line.id], lane_centerlines_dict), source[0], source[-1])

        source_ref_line = crop_line(build_line(source_ids, lane_centerlines_dict), source[0], source[-1])
        
        if angle<90:
            headstart = headstart[::-1]
            source_ref_line = source_ref_line[::-1]
        #endregion    
        
        #endregion
        
        if source_closest_line.id == target_closest_line.id:
            continue

        # Search lines
        lines = []

        # Source lines
        s_lines = [source_closest_line.centerline]

        # Target lines
        t_lines = []

        # Limit of search
        limit = 10

        # # Separation point
        # separation_points = []
            
        # Target closest line is found or not
        found = False

        # Go successors or predecessors
        front = angle>=90

        # Pandas dataframe to save lines
        df = pd.DataFrame(columns=["parent_id", "id"])

        # Next Lines
        lines = source_closest_line.successors if front else source_closest_line.predecessors

        if lines is None:
            continue

        df = df.append( [ {"parent_id":-1, "id":id}  for id in lines] , ignore_index=True)

        
        # Try limit times
        for j in range(limit):

            # Stop for 0 lanes
            if len(lines)==0:
                break
            
            # Check for target lane
            if target_closest_line.id in lines:
                found = True
                break
            
            # Traverse lines
            temp = lines
            lines = []
            for parent_id in temp:
                
                # Get Next Line indexes
                next_lines = lane_centerlines_dict[parent_id].successors if front else lane_centerlines_dict[parent_id].predecessors
                if next_lines is not None:
                    lines.extend(next_lines)
                    df = df.append( [ {"parent_id": parent_id, "id": id} for id in next_lines], ignore_index=True)               


        if found:

            # Remove if there is no intersection
            if (df.groupby(['parent_id']).count().max()[0])<=1:
                continue

            df, intersection_ids = df_line_filter(df, target_closest_line.id, lane_centerlines_dict)
            

            #region Build lines

            lines = [ [row["id"]] for _, row in df.loc[df["parent_id"]==-1].iterrows()]
            done = False

            while not done:
                done = True
                last = [l[-1] for l in lines]

                for j, parent_id in enumerate(last):

                    new_ids = [ row["id"] for _, row in df.loc[df["parent_id"]==parent_id].iterrows()]
                    
                    if len(new_ids)>0:
                        done = False
                        lines.extend( [lines[j] + [id] for id in new_ids[1::]] )
                        lines[j].append(new_ids[0])
            # Label


            
            label = [ line[-1] for line in lines].index(target_closest_line.id)
            
            lines = [ crop_line(line, source[-2], line[-1]) for line in [ headstart + build_line(line, lane_centerlines_dict, reverse=(not front)) for line in lines] ]
            
            # Label build
            label_line = crop_line( lines[label], source[-2], target[-1])
            label_line_distance = line_distance(label_line) + 2
            
            #endregion


            # # Distance threshold
            # lines = [ line_distance_crop(line, label_line_distance) for line in lines]

            #region Build intersection polygon

            intersection = Polygon(avm.get_lane_segment_polygon(intersection_ids[0], city)[:, 0:2])

            for id in intersection_ids[1:]:
                intersection = intersection.intersection(Polygon(avm.get_lane_segment_polygon(id, city)[:, 0:2]))

            #endregion

            #region Find Separation Point = centroid of intersection

            separation_point = find_separation_point( [build_line([id], lane_centerlines_dict, reverse=(not front)) for id in intersection_ids ] )

            #endregion



            # Check if target is outside intersection and (source is in intersection or source 2 TTC close to separation_point)
            if not Point(target[-1]).within(intersection) and ( Point(source[-1]).within(intersection) or before_separation_point(source, separation_point) ):
                
                ret.append( {"city" : city, "source_ref_line" : source_ref_line, "source_trj": source, "target_trj":target, "label": label, "candidate_cl": lines } )
                
    
    return ret


# thread_run(recs, "output/v4/train.pkl")

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

    