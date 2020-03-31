import math
import numpy as np
import pickle
import cv2
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
import itertools
from copy import copy

def vector_direction(vector):
    
    x, y = vector

    angle = np.arctan2(y, x)*180/np.pi

    return angle

def direction_difference(direction_1, direction_2):

    diff = direction_1 - direction_2

    if diff > 180:
        diff = diff - 360
    elif diff < -180:
        diff = diff + 360

    return abs(diff)
    

def load_pickle(file_name):
    file = open(file_name, "rb")
    data = pickle.load(file)
    file.close()
    return data


def save_pickle(data, file_name):

    file = open(file_name, "wb")
    pickle.dump(data, file)
    file.close() 

def save_pickle_async(data, filename):
    AsyncWrite(data, filename).start()

import threading

class AsyncWrite(threading.Thread):

    def __init__(self, data, filename):

        threading.Thread.__init__(self)
        self.data = data
        self.filename = filename
    
    def run(self):
        save_pickle(self.data, self.filename)

def df_line_filter(df, label_id, line_dict):

    #region Find first intersection id
    diversion_id = -1

    while True:
        lines = df.loc[df["parent_id"]==diversion_id]

        if len(lines)>1:
            break
        diversion_id = lines.iloc[0]["id"]

    #endregion

    #region Remove all neighbor lines from target line
    id = label_id

    while True:
        
        lines = df.loc[df["id"]==id]

        if len(lines)<=0:
            break

        parent_id = lines.iloc[0]["parent_id"]

        if parent_id==diversion_id:
            break

        df = df[ (df.parent_id!=parent_id) | (df.id==id) ]

        id = parent_id
    #endregion

    #region Remove unnecessary lines

    for _, row in df.loc[df["parent_id"]==diversion_id].iterrows():
        
        parent_id = row["id"]
        lines = df[ df["parent_id"]==parent_id ]

        while len(lines)>0:
            new_id = lines.iloc[0]["id"]

            # Check for no Turns
            for id in lines["id"].tolist():
                if line_dict[id].turn_direction=="NONE":
                    new_id = id
                    break

            df = df[ (df.parent_id!=parent_id) | (df.id==new_id) ]

            parent_id = new_id

            lines = df[ df["parent_id"]==parent_id ]
            
    #endregion

    intersection_ids = df[ df["parent_id"]==diversion_id ].id.tolist()

    return df, intersection_ids



#region Frechet Distance

def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = distance(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),distance(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),distance(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),distance(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""
def frechetDist(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)
#endregion

def draw_map(avm, center, city, image=None, img_size=1024):
    """
    Returns image of map area including Centerlines, Polygons
    """
    # Create Image
    if image==None:
        image = np.full((img_size, img_size, 3), 255, dtype='uint8')
    
    ratio = 8
    center_x, center_y = center
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2
    max_x, max_y = center_x + img_size/ratio/2, center_y + img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 


    #region Lane Polygons

  

    lane_polygons = avm.find_local_lane_polygons( [min_x, max_x, min_y, max_y], city )

    for lane in lane_polygons:

        lane = lane[:, 0:2].tolist()
        lane = c2p(lane)

        image = draw_line(image, lane, color=(66, 160, 237), width=1)
    

    #endregion

    #region CenterLines


    centerlines = avm.find_local_lane_centerlines( (min_x+max_x)/2, (min_y+max_y)/2, city, np.amax( [ (max_x-min_x)/2, (max_y-min_y)/2 ] ) )


    for line in centerlines:
    
        line = line[:, 0:2]
        line = c2p(line)

        image = draw_line(image, line, color=(5, 21, 125), width=1)

    #endregion
    
    return image
  



def draw_line(image, line, color, width):
    
    for i in range(len(line)-1):
        image = cv2.line(image, tuple(line[i]), tuple(line[i+1]), color, width)
    return image

def draw_2(source, target=[], s_line=[], t_line=[], label_line=[], other_lines=[], points=[], polygons=[], image=None, img_size=1024):

    if image is None:
        image = np.full((img_size, img_size, 3), 255, dtype='uint8')

    ratio = 8
    center_x, center_y = source[-1]
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 

    image = draw_line(image, c2p(source), color=(6, 196, 165), width=3)
    image = draw_line(image, c2p(target), color=(237, 17, 75), width=3)

    
   
    if len(s_line)>0:
        image = draw_line(image, c2p(s_line), color=(120, 145, 4), width=2)
        image = cv2.circle(image, c2p([s_line[0]])[0], 5, (120, 145, 4), -1)
        image = cv2.circle(image, c2p([s_line[-1]])[0], 5, (120, 145, 4), -1)

   

    if len(t_line) > 0:
        image = draw_line(image, c2p(t_line), color=(230, 11, 150), width=2)

        image = cv2.circle(image, c2p([t_line[0]])[0], 5, (230, 11, 150), -1)
        image = cv2.circle(image, c2p([t_line[-1]])[0], 5, (230, 11, 150), -1)

    for other_line in other_lines:
        
        if len(other_line)>0:
            image = draw_line(image, c2p(other_line), color=(121, 5, 171), width=2)

            # Draw circle in end points
            image = cv2.circle(image, c2p([other_line[0]])[0], 5, (37, 133, 3), -1)
            image = cv2.circle(image, c2p([other_line[-1]])[0], 5, (121, 5, 171), -1)

    if len(label_line) > 0:
    
        image = draw_line(image, c2p(label_line), color=(245, 97, 5), width=2)
        image = cv2.circle(image, c2p([label_line[0]])[0], 5, (245, 97, 5), -1)
        image = cv2.circle(image, c2p([label_line[-1]])[0], 5, (245, 97, 5), -1)

    

    

    for point in points:
        image = cv2.circle(image, c2p([point])[0], 4, (4, 186, 214), -1)

    for polygon in polygons:
        image = draw_line(image, c2p(polygon), color=(4, 138, 143), width=1)


    return image

def draw(source, target=[], s_line=[], t_line=[], label_line=[], other_lines=[], points=[], polygons=[], image=None, img_size=1024):
    
    if image is None:
        image = np.full((img_size, img_size, 3), 255, dtype='uint8')

    ratio = 8
    center_x, center_y = source[-1]
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 

    image = draw_line(image, c2p(source), color=(6, 196, 165), width=3)
    image = draw_line(image, c2p(target), color=(237, 17, 75), width=3)

    
   
    if len(s_line)>0:
        image = draw_line(image, c2p(s_line), color=(120, 145, 4), width=2)
        image = cv2.circle(image, c2p([s_line[0]])[0], 5, (120, 145, 4), -1)
        image = cv2.circle(image, c2p([s_line[-1]])[0], 5, (120, 145, 4), -1)

   

    if len(t_line) > 0:
        image = draw_line(image, c2p(t_line), color=(230, 11, 150), width=2)

        image = cv2.circle(image, c2p([t_line[0]])[0], 5, (230, 11, 150), -1)
        image = cv2.circle(image, c2p([t_line[-1]])[0], 5, (230, 11, 150), -1)

    for other_line in other_lines:
        
        if len(other_line)>0:
            image = draw_line(image, c2p(other_line), color=(3, 107, 252), width=2)

            # Draw circle in end points
            image = cv2.circle(image, c2p([other_line[0]])[0], 5, (3, 107, 252), -1)
            image = cv2.circle(image, c2p([other_line[-1]])[0], 5, (3, 107, 252), -1)

    if len(label_line) > 0:
    
        image = draw_line(image, c2p(label_line), color=(245, 97, 5), width=2)
        image = cv2.circle(image, c2p([label_line[0]])[0], 5, (245, 97, 5), -1)
        image = cv2.circle(image, c2p([label_line[-1]])[0], 5, (245, 97, 5), -1)

    

    

    for point in points:
        image = cv2.circle(image, c2p([point])[0], 4, (4, 186, 214), -1)

    for polygon in polygons:
        image = draw_line(image, c2p(polygon), color=(4, 138, 143), width=1)


    return image

def draw(source, target=[], s_line=[], t_lines=[], label_line=[], other_lines=[], points=[], polygons=[], image=None, img_size=1024):
    
    if image is None:
        image = np.full((img_size, img_size, 3), 255, dtype='uint8')

    ratio = 8
    center_x, center_y = source[-1]
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 

    for other_line in other_lines:
        
        if len(other_line)>0:
            image = draw_line(image, c2p(other_line), color=(3, 107, 252), width=2)

            # Draw circle in end points
            image = cv2.circle(image, c2p([other_line[0]])[0], 5, (3, 107, 252), -1)
            image = cv2.circle(image, c2p([other_line[-1]])[0], 5, (3, 107, 252), -1)

    image = draw_line(image, c2p(source), color=(6, 196, 165), width=3)
    # image = cv2.circle(image, c2p([source[0]])[0], 5, (255, 0, 0), -1)
    # image = cv2.circle(image, c2p([source[-1]])[0], 5, (0, 0, 255), -1)



    image = draw_line(image, c2p(target), color=(237, 17, 75), width=3)

   

    for t_line in t_lines:
        image = draw_line(image, c2p(t_line), color=(121, 4, 189), width=2)

        image = cv2.circle(image, c2p([t_line[0]])[0], 5, (121, 4, 189), -1)
        image = cv2.circle(image, c2p([t_line[-1]])[0], 5, (121, 4, 189), -1)

    if len(label_line) > 0:
    
        image = draw_line(image, c2p(label_line), color=(120, 145, 4), width=2)
        image = cv2.circle(image, c2p([label_line[0]])[0], 5, (120, 145, 4), -1)
        image = cv2.circle(image, c2p([label_line[-1]])[0], 5, (120, 145, 4), -1)


    if len(s_line)>0:
        image = draw_line(image, c2p(s_line), color=(31, 13, 224), width=2)
        image = cv2.circle(image, c2p([s_line[0]])[0], 5, (31, 13, 224), -1)
        image = cv2.circle(image, c2p([s_line[-1]])[0], 5, (31, 13, 224), -1)


    

    

    for point in points:
        image = cv2.circle(image, c2p([point])[0], 4, (4, 186, 214), -1)

    for polygon in polygons:
        image = draw_line(image, c2p(polygon), color=(4, 138, 143), width=1)


    return image

def draw_string(image, text, location):
    
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # fontScale 
    fontScale = 0.5
    # Blue color in BGR 
    color = (255, 0, 0)
    thickness = 1

    image = cv2.putText(image, text, location, font, fontScale, color, thickness, cv2.LINE_AA)

    return image

def draw_with_scores(source, target=[], s_line=[], t_line=[], label_line=[], other_lines=[], points=[], polygons=[], scores=[], locations=[], image=None, img_size=1024):
    
    if image is None:
        image = np.full((img_size, img_size, 3), 255, dtype='uint8')

    ratio = 8
    center_x, center_y = source[-1]
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 

    image = draw_line(image, c2p(source), color=(6, 196, 165), width=2)
    image = draw_line(image, c2p(target), color=(5, 30, 171), width=2)

    
   
    if len(s_line)>0:
        image = draw_line(image, c2p(s_line), color=(120, 145, 4), width=2)
        image = cv2.circle(image, c2p([s_line[0]])[0], 5, (120, 145, 4), -1)
        image = cv2.circle(image, c2p([s_line[-1]])[0], 5, (120, 145, 4), -1)

   

    if len(t_line) > 0:
        image = draw_line(image, c2p(t_line), color=(230, 11, 150), width=2)

        image = cv2.circle(image, c2p([t_line[0]])[0], 5, (230, 11, 150), -1)
        image = cv2.circle(image, c2p([t_line[-1]])[0], 5, (230, 11, 150), -1)

    for other_line in other_lines:
    
        image = draw_line(image, c2p(other_line), color=(121, 5, 171), width=2)

        # Draw circle in end points
        image = cv2.circle(image, c2p([other_line[0]])[0], 5, (37, 133, 3), -1)
        image = cv2.circle(image, c2p([other_line[-1]])[0], 5, (121, 5, 171), -1)

    if len(label_line) > 0:
    
        image = draw_line(image, c2p(label_line), color=(245, 97, 5), width=2)
        image = cv2.circle(image, c2p([label_line[0]])[0], 5, (245, 97, 5), -1)
        image = cv2.circle(image, c2p([label_line[-1]])[0], 5, (245, 97, 5), -1)

    

    

    for point in points:
        image = cv2.circle(image, c2p([point])[0], 4, (4, 186, 214), -1)

    for polygon in polygons:
        image = draw_line(image, c2p(polygon), color=(4, 138, 143), width=1)


    locations = c2p(locations)

    for score, location in zip(scores, locations):
        image = draw_string(image, score, location)


    return image


def draw_scores(center, scores=[], locations=[], image=None, img_size=1024):
    
    if image is None:
        image = np.full((img_size, img_size, 3), 0, dtype='uint8')

    ratio = 8
    center_x, center_y = center
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 

    locations = c2p(locations)

    for score, location in zip(scores, locations):
        image = draw_string(image, score, location)


    return image

def draw_lines(lines, center, img_size=1024):

    image = np.full((img_size, img_size, 3), 255, dtype='uint8')

    ratio = 8
    center_x, center_y = center
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 

    for line in lines:
        image = draw_line(image, c2p(line), color=(5, 21, 125), width=1)

    return image

def angle_between_lines(line_a, line_b):

    line_a = np.asarray(line_a)
    line_b = np.asarray(line_b)

    # Move to center
    a = line_a[-1] - line_a[0]
    b = line_a[-1] - line_b[0]

    # Find slopes
    slope_a = a[1]/a[0]
    slope_b = b[1]/b[0]

    return math.degrees(math.pi - abs( math.atan(slope_a) - math.atan(slope_b) ) )


def hausdorffDistance(line1, line2):
    

    x_size = len(line1)
    y_size = len(line2)


    # Calculate distances from points
    distArrOne = np.zeros( (x_size, y_size) )

    for x in range(x_size):
        for y in range(y_size):
            distArrOne[x][y] = distance(line1[x], line2[y])
    
    distArrTwo = np.swapaxes(distArrOne, 0, 1)


    ##Finally calculates Hausdorff Distance
    #Calculate distances between origin and target feature
    H1 = max([min([distArrOne[i][j] for i in range(x_size)]) for j in range(y_size)])  #get the highest minimum (supremum infimum) travelling along axis 1 (y-axis)
    H2 = max([min([distArrOne[i][j] for j in range(y_size)]) for i in range(x_size)])  #get the highest minimum (supremum infimum) travelling along axis 0 (x-axis)
    #print H1, H2
    #Repeat the calculation in reverse order
    H3 = max([min([distArrTwo[j][i] for i in range(x_size)]) for j in range(y_size)])  #get the highest minimum (supremum infimum) travelling along axis 1 (y-axis)
    H4 = max([min([distArrTwo[j][i] for j in range(y_size)]) for i in range(x_size)])  #get the highest minimum (supremum infimum) travelling along axis 0 (x-axis)
    #print H3, H4

    hausdorff = max([H1, H2]+[H3, H4])
    #print hausdorff

    return hausdorff


def find_min_max_xy(points):
    
    points = np.asarray(points)
    x_min, y_min = np.amin(points[:, 0]), np.amin(points[:, 1])
    x_max, y_max = np.amax(points[:, 0]), np.amax(points[:, 1])
    return x_min, y_min, x_max, y_max

from shapely.geometry import Point, LineString

def find_closest_centerline(avm, points, city):

    # Frechet
    x_min, y_min, x_max, y_max = find_min_max_xy(points)
    x_mid, y_mid = (x_min+x_max)/2, (y_min+y_max)/2

    manhattan_range = distance((x_min, y_min), (x_mid, y_mid))

    lane_ids = avm.get_lane_ids_in_xy_bbox(x_mid, y_mid, city, query_search_range_manhattan=manhattan_range)

    # Centerline dictionary
    lane_centerlines_dict = avm.city_lane_centerlines_dict[city]

    lanes = [ LineString(lane_centerlines_dict[id].centerline) for id in lane_ids]

    points = [Point(p) for p in points]

    distances = [ np.sum( [ p.distance(lane) for p in points] )  for lane in lanes]

    index = np.argmin(distances)    

    return lane_ids[index], (lane_ids[:index]+ lane_ids[index+1:])

    # Ordinary
    # # IDs and occurence counts
    # ids, counts = np.unique([avm.get_nearest_centerline(p, city)[0].id for p in points], return_counts=True)
    
    # # Biggest occurence
    # index = np.argmax(counts)
    # return ids[index]

def build_line(ids, line_dict, reverse=False):

    line = []
    for id in ids:

        if not reverse:
            line = line + [ tuple(p) for p in  line_dict[id].centerline]
        else:
            line = line + [ tuple(p) for p in  line_dict[id].centerline[::-1]]

    return line


def sublist(lst, ref_lst):
    """
    Returns True if lst is sublist of ref_lst
    """
    ls = [i for i, el in enumerate(lst) if el in ref_lst]

    if len(ls)!=len(lst):
        return False

    for i in range(1, len(ls)):
        if i!=ls[i]-ls[0]:
            return False
    
    return True
    

def combine_lines(line_ids, line_dict):
    """
    Combine all lines within line_ids

    Args:
        line_ids:   List of line ids
        line_dict:  Correct id to line dictinary from ArgoverseMap 
    """
    lines = []

    # while len(line_ids)>0:
    for i in range(len(line_ids)):

        cand_id = line_ids[i]
        # del line_ids[0]

        cand_lines = [ [cand_id] ]

        done = False
        while not done:
            done = True
            # remove_items = []
            
            for j in range(len(cand_lines)):
                # Predecessors and successors
                predecessors = line_dict[cand_lines[j][0]].predecessors
                successors = line_dict[cand_lines[j][-1]].successors

                # Check for valid predecessors
                if predecessors is not None:
                    
                    predecessors = sorted( [ p for p in predecessors if p in line_ids and p not in cand_lines[j]], key=lambda p: line_dict[p].turn_direction=="NONE", reverse=True)

                    if len(predecessors)>0:
                        p = predecessors[0]
                        # When multiple lines create new branches
                        if len(predecessors) > 1:
                            cand_lines.extend( [ ([p] + cand_lines[j]) for p in predecessors[1:] ])

                        cand_lines[j] = [p] + cand_lines[j]
                        # remove_items.append(p)

                        done = False

                # Check for valid successors
                if successors is not None:
                    successors = sorted([ s for s in successors if s in line_ids and s not in cand_lines[j]], key=lambda s: line_dict[s].turn_direction=="NONE", reverse=True )

                    if len(successors)>0:
                        s = successors[0]

                        # When multiple lines create new branches
                        if len(successors) > 1:
                            cand_lines.extend( [ (cand_lines[j] + [s]) for s in successors[1:] ])

                        cand_lines[j] = cand_lines[j] + [s]
                        # remove_items.append(s)
                        done = False

            # #region Remove items
            # for item in np.unique(remove_items):
            #     line_ids.remove(item)
            # #endregion
        
        # Add new candidate lines
        lines.extend(cand_lines)

    final = []

    #region Hopefully eliminates duplicates or sublists
    for line in lines:
        if line not in final and line[::-1] not in final:
            final.append(line)
        else:
            for i, f in enumerate(final):
                if len(line) > len(f) and sublist(f, line):
                    del final[i]
                    final.append(line)
                    break
    #endregion

    lines = final

    return lines

#region New Additions
def unique(lst):
    final = []

    for el in lst:
        if not any( sublist(el, f) for f in final):
            final.append(el)
    return final

def line_distance_crop(line, threshold):
    """
    Crop line with given distance threshold
    """
    
    total_length = 0
    for i in range(1, len(line)):
        total_length = total_length + distance(line[i], line[i-1])
        if total_length>=threshold:
            return line[:i+1]
    return line

def line_distance(line):
    """
    Returns distance of given line: list of points
    """
    return np.sum( [ distance(line[i], line[i+1]) for i in range(len(line)-1)] )


def random_color():
    
    return ( np.random.randint(0, 256), np.random.randint(0, 256),  np.random.randint(0, 256)) 


def draw_lines_rand_color(lines, center, img_size=1024):
    
    image = np.full((img_size, img_size, 3), 255, dtype='uint8')

    ratio = 8
    center_x, center_y = center
    min_x, min_y = center_x - img_size/ratio/2, center_y - img_size/ratio/2

    # Coordinate to pixel convert function
    c2p = lambda line: [ ( int((p[0] - min_x)*ratio), int((p[1] - min_y)*ratio)) for p in line ] 

    for line in lines:
        color = random_color()
        image = draw_line(image, c2p(line), color=color, width=1)

        image = cv2.circle(image, c2p([line[0]])[0], 5, color, -1)
        image = cv2.circle(image, c2p([line[-1]])[0], 5, color, -1)

        cv2.imshow("image", image)
        cv2.waitKey(1000)

    cv2.waitKey(1000)

    return image

def crop_line(line, start, end, extended=False):
    """
    Returns cropped line by finding closest points 
    in line to start and end
    """

    # Find closest point to start point
    distances = [ distance(start, p) for p in line]
    start_index = np.argmin(distances)


    # Find closest point to end point
    distances = [ distance(end, p) for p in line]
    end_index = np.argmin(distances)

    #region Slice line
    step = -1 if end_index<start_index else 1

    if start_index == end_index:
        if start_index == 0:
            if extended:
                return line[0:2], 0, 2
            return line[0:2]
        
        start_index = start_index + -1 * step

    elif end_index == 0:
        if extended:
            return line[start_index: None: step], start_index, end_index
        return line[start_index: None: step]
    
    #region Extended return
    if extended:
        return line[start_index: end_index+step: step], start_index, end_index
    #endregion


    return line[start_index: end_index+step: step]
    #endregion

def crop_line_ids(ids, line_dict, start, end):
    """
    Returns sliced line ids that contain start_coord and end_coord
    """
    # Find start and end coordinate indexes
    _, start_coord, end_coord = crop_line( build_line(ids, line_dict), start, end, extended=True)

    if start_coord > end_coord:
        # Swap
        t = end_coord
        end_coord = start_coord
        start_coord = t


    # Line coordinates lengths
    line_lengths = [ len(line_dict[id].centerline) for id in ids ]

    s_index = -1
    e_index = -1
    
    sum = 0
    
    for i, length in enumerate(line_lengths):
        
        if s_index!=-1 and e_index!=-1:
            break
        
        # find s_index
        if s_index==-1 and sum<=start_coord and sum+length>start_coord:
            s_index = i
        
        # find e_index
        if e_index==-1 and sum<=end_coord and sum+length>end_coord:
            e_index = i
        
        sum = sum + length

    return ids[s_index: e_index+1]

def find_closest_centerline_v2(avm, line, city, additional=[], from_back=True):
    """
    Returns id and additional coordinates tuple(id, list)

    """
    if from_back:
        points = line + additional
    else:
        points = additional + line

    # hausdorff
    x_min, y_min, x_max, y_max = find_min_max_xy(points)
    x_mid, y_mid = (x_min+x_max)/2, (y_min+y_max)/2

    manhattan_range = distance((x_min, y_min), (x_mid, y_mid))
    # Set threshold
    manhattan_threshold = 5
    if manhattan_range < manhattan_threshold:
        manhattan_range = manhattan_threshold

    line_ids = avm.get_lane_ids_in_xy_bbox(x_mid, y_mid, city, query_search_range_manhattan=manhattan_range)

    # Centerline dictionary
    line_dict = avm.city_lane_centerlines_dict[city]

    # Find combined lines
    line_ids = combine_lines(line_ids, line_dict)

    # Make line into tuple of  (point array, ids)
    line_ids = [ ( crop_line(build_line(ids, line_dict), line[0], line[-1]), ids)  for ids in line_ids]

    # Calculate distances
    distances = [ (hausdorffDistance(coords, points), line_distance(coords) )  for coords, ids in line_ids]

    # Find index with smallest distance
    index = sorted(range(len(distances)), key=lambda k: distances[k])[0]

    # ID list of centerlines
    ids = crop_line_ids( line_ids[index][1], line_dict, line[0], line[-1])

    return ids
#endregion



def rotate(point, ref, angle, clockwise=False):
    """
    Rotate point around ref at angle
    """

    # sine and cosine
    sin = math.sin(math.radians(angle))
    cos = math.cos(math.radians(angle))

    cx, cy = ref
    px, py = point

    # translate point back to origin
    px = px - cx
    py = py - cy

    # new point
    if clockwise:
        nx = px * cos + py * sin
        ny = -px * sin + py * cos
    else:
        nx = px * cos - py * sin
        ny = px * sin + py * cos

    # return to ref point
    px = nx + cx
    py = ny + cy

    return px, py

def angle_y_axis(point, ref):
    """
    Calculate angle line between y-axis
    """

    x1, y1 = ref
    x2, y2 = point

    x_sign = (x2 > x1)

    y_sign = (y2 > y1)


    if x2 > x1:
        # Swap x1 and x2
        x1, x2 = x2, x1

    if y2 > y1:
        # Swap y1 and y2
        y1, y2 = y2, y1

    
    angle = math.degrees( math.acos( (y1-y2)/ math.sqrt( (x1-x2)**2 + (y1-y2)**2 ) ) )

    #region 4 Quarters
    
    if x_sign and y_sign:
        # I quarter
        angle = 180 + angle
    
    elif not x_sign and y_sign:
        # II quarter
        angle = 180 - angle
    
    elif x_sign and not y_sign:
        # IV quarter
        angle = 360 - angle

    #endregion

    return angle

def crossed_separation_point(line, point, threshold=0.5):

    point = (point[0] - line[0][0], point[1] - line[0][1])
    # Center reference
    line_start = (0, 0)
    line_end = line[-1] - line[0]

    # Calculate Angle 
    angle = angle_y_axis(line_end, line_start)

    # Rotate Angle
    line_end = rotate(line_end, line_start, angle)
    point = rotate(point, line_start, angle) 
    
    return line_end[1]+threshold<= point[1]

def before_separation_point(line, point, times=20):

    threshold = np.average([ distance(line[i], line[i+1]) for i in range(len(line)-1)]) * times

    point = (point[0] - line[0][0], point[1] - line[0][1])
    # Center reference
    line_start = (0, 0)
    line_end = (line[-1][0] - line[0][0], line[-1][1] - line[0][1])

    # Calculate Angle 
    angle = angle_y_axis(line_end, line_start)

    # Rotate Angle
    line_end = rotate(line_end, line_start, angle)
    point = rotate(point, line_start, angle) 

    return  line_end[1]>point[1] and line_end[1]-threshold<= point[1]




def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)

def find_separation_point(lines):

    points = []

    threshold = 0.2

    for i in range(len(lines)-1):

        ref_line = LineString(lines[i])
        line = lines[i+1]

        closest_points = sorted([ (distance(p1, (p2.x, p2.y)), p1, (p2.x, p2.y) )  for p1, p2 in [ (p, nearest_points(ref_line, Point(p))[0] ) for p in line] ], key=lambda x: x[0] )

        for j, (dist, p1, p2) in enumerate(closest_points):

            if dist>threshold or j==len(closest_points)-1:
                points.append( ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2) )
                break
    
    return ( np.average([ p[0] for p in points]), np.average([ p[1] for p in points]) )



from argoverse.utils.centerline_utils import (
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    lane_waypt_to_query_dist,
    remove_overlapping_lane_seq,
)

def nfs(avm, id, city_name, look_for_left=True, look_for_right=True, call=1):
    """
    Neighbor first search - kind of DFS
    Search for neighbors in a same direction
    """

    # print("ID: %d, look_for_left:%d, look_for_right:%d Call: %d"%(id, look_for_left, look_for_right, call))

    # Current lane information
    curr_lane = avm.city_lane_centerlines_dict[city_name][id]

    # List of lanes to return
    lanes = []

    # Check function to identify if lane is in same direction
    check = lambda ref_lane, neighbor: distance(ref_lane[0], neighbor[0])<=distance(ref_lane[-1], neighbor[0]) and distance(ref_lane[0], neighbor[-1])>=distance(ref_lane[-1], neighbor[-1])

    # Check right lane
    neighbor_id = curr_lane.r_neighbor_id
    if look_for_right and neighbor_id is not None and check(curr_lane.centerline, avm.get_lane_segment_centerline(neighbor_id, city_name)):
        lanes.extend(nfs(avm, neighbor_id, city_name, look_for_left=False, call=call+1))
        lanes.append(neighbor_id)
    
    # Check left lane
    neighbor_id = curr_lane.l_neighbor_id
    if look_for_left and neighbor_id is not None and check(curr_lane.centerline, avm.get_lane_segment_centerline(neighbor_id, city_name)):
        lanes.extend(nfs(avm, neighbor_id, city_name, look_for_right=False, call=call+1))
        lanes.append(neighbor_id)

    return lanes



def displacement_vector(line):

    length = len(line)
    if length<=1:
        raise ValueError('Line must have at least 2 points')

    
    displacements = np.asarray([ (line[i+1][0]-line[i][0], line[i+1][1]-line[i][1])  for i in range(length-1)] )

    return ( np.average(displacements[:, 0])/(length-1), np.average(displacements[:, 1])/(length-1) ) 


def angle_between(v1, v2):

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def shortest_distance(line1, line2):

    line_string_1 = LineString(line1)
    line_string_2 = LineString(line2)
    
    return np.amin( [ line_string_2.distance(Point(p)) for p in line1] + [ line_string_1.distance(Point(p)) for p in line2] )

def process_line(avm, ref_point, line, city_name, front=50, back=30):
    """
    Cut from given point and cut in front "front" distance
    and cut in back "back" distance
    """

    segment_id, inside_id = -1, -1
    min_distance = math.inf

    #region Select min
    
    for i, (id, points) in enumerate(line):

        for j, point in enumerate(points):

            
            curr_distance = distance(point, ref_point)
            if curr_distance < min_distance:
                segment_id = i
                inside_id = j
                min_distance = curr_distance
       
    #endregion

    ret = []
    front_done = False
    back_done = False

    
    #region Back search

    # Current point
    curr_segment_id = segment_id
    curr_inside_id = inside_id

    curr_point = line[curr_segment_id][1][curr_inside_id]

    #region Search from current segment
    for point in line[segment_id][1][curr_inside_id-1: : -1]:

        back = back - distance(curr_point, point)
        curr_point = point
        ret = [point] + ret

        if back<=0:
            back_done = True
            break

        
    #endregion

    #region Search from other segments
    if not back_done and curr_segment_id!=0:
       
        for id, points in line[curr_segment_id-1: :-1]:
          
            if back_done:
                break

            for point in points[::-1]:

                back = back - distance(curr_point, point)
                curr_point = point
                ret = [point] + ret
                if back<=0:
                    back_done = True
                    break
    #endregion
    
    #region Search outside line
    curr_segment_id = line[0][0]
    while not back_done:

        ids = avm.get_lane_segment_predecessor_ids(curr_segment_id, city_name)
        if ids is None:
            back_done = True
            break

        curr_segment_id = None
        for id in ids:
            if avm.get_lane_turn_direction(id, city_name)=='NONE':
                curr_segment_id = id
        
        curr_segment_id =  curr_segment_id if curr_segment_id!=None else ids[0]

        for point in avm.get_lane_segment_centerline(curr_segment_id, city_name)[::-1, :2]:

            back = back - distance(curr_point, point)
            curr_point = point
            ret = [point] + ret
            if back<=0:
                back_done = True
                break


            

    # #endregion


    #endregion


    #region Front search

    # Current point
    curr_segment_id = segment_id
    curr_inside_id = inside_id

    curr_point = line[curr_segment_id][1][curr_inside_id]

    #region Search from current segment
    for point in line[segment_id][1][curr_inside_id+1:]:

        front = front - distance(curr_point, point)
        curr_point = point
        ret = ret + [point] 

        if front<=0:
            front_done = True
            break

        
    #endregion

    #region Search from other segments
    if not front_done:
       
        for id, points in line[curr_segment_id+1:]:
          
            if front_done:
                break

            for point in points:

                front = front - distance(curr_point, point)
                curr_point = point
                ret = ret + [point]
                if front<=0:
                    front_done = True
                    break
    #endregion
    
    #region Search outside line
    curr_segment_id = line[-1][0]
    while not front_done:

        ids = avm.get_lane_segment_successor_ids(curr_segment_id, city_name)
        if ids is None:
            front_done = True
            break

        curr_segment_id = None
        for id in ids:
            if avm.get_lane_turn_direction(id, city_name)=='NONE':
                curr_segment_id = id
        
        curr_segment_id =  curr_segment_id if curr_segment_id!=None else ids[0]

        for point in avm.get_lane_segment_centerline(curr_segment_id, city_name)[:, :2]:

            front = front - distance(curr_point, point)
            curr_point = point
            ret = ret + [point]
            if front<=0:
                front_done = True
                break
            

    #endregion


    #endregion


    return ret


def closest_line(lines, traj):
    # """
    # Select most frequent line
    # """
    
    # lines = [ LineString(line) for line in lines]

    # lines_count = len(lines)

    # frequency = [ min(range(lines_count), key=lambda k : lines[k].distance(Point(p)) ) for p in traj]

    # index, count = np.unique(frequency, return_counts=True)

    # return index[np.argmax(count)]  


    lines = [ LineString(line) for line in lines]
    points = [ Point(p) for p in traj]
    distances = [ np.average([ line.distance(p) for p in points]) for line in lines ]

    return min( range(len(lines)), key=lambda k: distances[k])





def get_candidate_centerlines(avm, traj, city_name):

    """
    Returns nearby closest centerlines to given trajectory
    """
    #region ROI

    roi_center = traj[-1]
    roi_radius = 20

    roi = Point(roi_center).buffer(roi_radius)

    #endregion

    line_ids = avm.get_lane_ids_in_xy_bbox(roi_center[0], roi_center[1], city_name, query_search_range_manhattan=roi_radius)

    traj_direction = vector_direction(displacement_vector(traj))
    

    line_ids = [id for id in line_ids if any([ roi.contains(Point(p)) for p in avm.get_lane_segment_centerline(id, city_name)]) ]

    lines = [ [ (id, avm.get_lane_segment_centerline(id, city_name)[:, :2] ) for id in ids] for ids in combine_lines(line_ids, avm.city_lane_centerlines_dict[city_name]) ]

    lines = [ process_line(avm, traj[-1], line, city_name)  for line in lines]

    # Distance cutting
    distance_threshold = 15
    lines = [ line for line in lines if len(line)>1 and LineString(line).distance(Point(traj[-1]))<distance_threshold] 
    

    limit = 6

    lines = sorted(lines, key = lambda line: ( LineString(line).distance(Point(traj[-1])), direction_difference(traj_direction, vector_direction( displacement_vector(line) ) ))  )[:limit]


    return lines, roi.exterior.coords

def get_candidate_centerlines_2(avm, traj, city_name):
    """
    Returns nearby closest centerlines to given trajectory
    """
    # Find reference lane to whole trajectory
    line_ids = find_closest_centerline_v2(avm, traj.tolist(), city_name)

    final_ids = []

    final_ids.extend(line_ids)

    #region DFS search
    dfs_threshold = max(20, 1.5 * line_distance(traj) )

    # # Second last point of source trajectory
    # point = Point(traj[18])

    # index = sorted(range(len(line_ids)), key=lambda k: LineString(avm.get_lane_segment_centerline(line_ids[k], city_name)).project(point))[0]

    index = 0

    for line in line_ids[index:]:

        for ids in avm.dfs(line, city_name, threshold=dfs_threshold):
            final_ids.extend(ids)
    

    #endregion

    final_ids = np.unique(final_ids).tolist()
    line_ids = copy(final_ids)

    #region Neighbor search
    
    

    for i, id in enumerate(line_ids):
        neighbors = nfs(avm, id, city_name)
        final_ids.extend(neighbors)

        for neighbor in neighbors:
            for ids in avm.dfs(neighbor, city_name, threshold=5):
                final_ids.extend(ids)
        
    #endregion



    final_ids = np.unique(final_ids)
    combined = combine_lines(final_ids, avm.city_lane_centerlines_dict[city_name])
    
    centerlines = avm.get_cl_from_lane_seq(combined, city_name)

    last_point = Point(traj[-1])
    
    limit = 6

    indexes = sorted(range(len(centerlines)), key=lambda k: ( LineString(centerlines[k]).distance(last_point), hausdorffDistance(traj.tolist(), centerlines[k].tolist()) ))[:limit]

    centerlines = [ centerlines[i] for i in range(len(centerlines)) if i in indexes]

    # label = sorted(range(len(centerlines)), key=lambda k: ( LineString(centerlines[k]).distance(last_point), hausdorffDistance(target_traj, centerlines[k].tolist()) ))[0]

    # return centerlines, label

    return centerlines


    





    


        






 