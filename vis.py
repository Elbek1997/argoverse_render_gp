from utils import *
from tqdm import tqdm
import cv2

from argoverse.map_representation.map_api import ArgoverseMap
avm = ArgoverseMap()

input_file = "val_results.pkl"

output_file = "vis/gp_results/%d.jpg"

def curv(line):

    ref = LineString([line[0], line[-1]])
    
    vector = np.asarray( [ ref.distance(Point(p)) for p in line] )
    
    return np.std(vector)


data = load_pickle(input_file)

for i in tqdm(range(len(data))):

    rec = data[i]

    source = rec["source_trj"]
    target = rec["target_trj"]

    ref_lanes = rec["candidate_cl"]
    length = len(ref_lanes)
    city = rec["city"]
    scores = rec["gp_result"]
    label = rec["label"]

    index = np.argmax(scores)
    other_lines = ref_lanes[:index] + ref_lanes[index+1:]
    label_line = ref_lanes[index]

    # print(curv(label_line))

    if label==index and curv(label_line)>=5:

        # Scores and locations
        score_strings = []
        locations = []

        for score, lane in zip(scores[: length], ref_lanes):
            score_strings.append("%.2f"%score)
            locations.append(lane[-1])

            

        image = draw_map(avm, source[-1], city)

        image = draw(source, target, s_line=label_line, image=image, other_lines=other_lines)

        image = draw_scores(source[-1], scores=score_strings, locations=locations, image=image)

        # image = cv2.flip(image, 1)

        cv2.imwrite(output_file%(i+1), image)

