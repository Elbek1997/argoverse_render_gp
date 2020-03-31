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
dataset = "input/train_raw.pkl"
# Output
output_file = "output/val.pkl"
 
print("[INFO]Loading Map...")
avm = ArgoverseMap()

city_lane_centerlines_dict = dc(avm.city_lane_centerlines_dict)


print("[INFO]Loading Dataset %s"%dataset)

limit = None
recs = pkl.load(open(dataset, "rb"))[144944:]
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

        source_ref_line = crop_line(build_line(source_ids, lane_centerlines_dict), source[0], source[-1])
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

        # Separation point
        separation_points = []

        # Separation centerlines
        separation_ids = []
            
        # Label index of target lane
        label = -1


        if angle >=90:
            # Go successors
            if source_closest_line.successors is None:
                continue
            lines = [ [id] for id in source_closest_line.successors]

            if len(lines)>1:
                separation_points.append(find_separation_point(lane_centerlines_dict[lines[0][-1]].centerline, lane_centerlines_dict[lines[1][-1]].centerline))
                separation_ids.append( [line[-1] for line in lines]) 
            # Try limit times
            for j in range(limit):

                # Stop for 0 lanes
                if len(lines)==0:
                    break
                
                # Check for target lane
                last = np.asarray( [ line[-1] for line in lines] )
                results = np.flatnonzero(last==target_closest_line.id)
                if len(results)>0:
                    label = results[0]
                    break

                
                # Traverse lines
                for index in range(len(lines)):
                    
                    # Get Next Line indexes
                    next_lines = lane_centerlines_dict[ lines[index][-1] ].successors
                    if next_lines is not None:

                        # Separation point
                        if len(next_lines)>1:
                            
                            separation_points.append(find_separation_point(lane_centerlines_dict[next_lines[0]].centerline, lane_centerlines_dict[next_lines[1]].centerline))
                            separation_ids.append(next_lines)
                            
                            # Add newly added lines
                            lines.extend( [ lines[index]+[next_line]  for next_line in next_lines[1:] ] )

                        # Append next_line
                        lines[index].append(next_lines[0])


            # Build lines 
            if label!=-1 and len(lines)>1:
                
                candidate_cl = []
                for line in lines:
                    line_form = []
                    for id in line:
                        line_form = line_form + [ tuple(p) for p in lane_centerlines_dict[id].centerline]
                        
                    candidate_cl.append(line_form)

        else:
            # Go predecessors
            # Reverse ref_line
            source_ref_line = source_ref_line[::-1]
            if source_closest_line.predecessors is None:
                continue
            lines = [ [id] for id in source_closest_line.predecessors]
            if len(lines)>1:
                separation_points.append(find_separation_point(lane_centerlines_dict[lines[0][-1]].centerline, lane_centerlines_dict[lines[1][-1]].centerline))
                separation_ids.append( [line[-1] for line in lines])
                            
            # Try limit times
            for j in range(limit):

                # Stop for 0 lanes
                if len(lines)==0:
                    break
                
                # Check for target lane
                last = np.asarray( [ line[-1] for line in lines] )
                results = np.flatnonzero(last==target_closest_line.id)
                if len(results)>0:
                    label = results[0]
                    break
                
                # Traverse lines
                for index in range(len(lines)):
                    
                    # Get Next Line indexes
                    next_lines = lane_centerlines_dict[ lines[index][-1] ].predecessors
                    if next_lines is not None:

                        # Separation point
                        if len(next_lines)>1:

                            separation_points.append(find_separation_point(lane_centerlines_dict[next_lines[0]].centerline, lane_centerlines_dict[next_lines[1]].centerline))
                            separation_ids.append(next_lines)
                            # Add newly added lines
                            lines.extend( [ lines[index]+[next_line]  for next_line in next_lines[1:] ] )
                            
                        # Append next_line
                        lines[index].append(next_lines[0])

            # Build lines 
            if label!=-1 and len(lines)>1:
                
                candidate_cl = []
                for line in lines:
                    line_form = []
                    for id in line:
                        line_form = line_form + [ tuple(p) for p in lane_centerlines_dict[id].centerline[::-1] ]

                    candidate_cl.append(line_form)

                
            
        
        if label!=-1 and len(lines)==2:

            #region Line Correction
            label_line = crop_line(candidate_cl[label], source[-1], target[-1])
            label_line_distance = line_distance(label_line)
            
            # Clear candidate_cls
            del candidate_cl[label]
            candidate_cl = unique([ line_distance_crop(crop_line(line, source[-1], line[-1]), label_line_distance) for line in candidate_cl])

            candidate_cl.append(label_line)
            label = len(candidate_cl)-1
            #endregion

            
            intersections = []

            for ids in separation_ids:
                intersection = Polygon(avm.get_lane_segment_polygon(ids[0], city)[:, 0:2])

                for id in ids[1:]:
                    intersection = intersection.intersection(Polygon(avm.get_lane_segment_polygon(id, city)[:, 0:2]))

                intersections.append(intersection)

            # intersections = [intersection.exterior.coords for intersection in intersections]
            intersection = intersections[0]
            separation_point = separation_points[0]
            
            if Point(source[-1]).within(intersection) or before_separation_point(source, separation_point):
            
                ret.append( {"city" : rec["city"], "source_trj": source, "target_trj":target, "label": label, "centerline": source_ref_line + label_line } )
            
    
    return ret


# thread_run(recs, "output/v2/train.pkl")

n_jobs = 8
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

    