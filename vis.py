from utils import *
from tqdm import tqdm
import cv2

from argoverse.map_representation.map_api import ArgoverseMap
avm = ArgoverseMap()

input_file = "result_pickle_final_test.pkl"

output_file = "vis/results_2/%d.jpg"


data = load_pickle(input_file)

for i in tqdm(range(len(data))):

    rec = data[i]

    source = rec["source"]
    # target = rec["target"]

    candidate_count = int(rec["num_cand"].item())
    if candidate_count==0:
        print(0)

    city = rec["city"]
    
    candidate_lanes = rec["ref_lane"].tolist()[:candidate_count]
    scores = rec["gp_result"][:candidate_count]

    predictions = rec["prediction"][:candidate_count]

    

    # Scores and locations
    score_strings = []
    locations = []

    for score, lane in zip(scores[: len(candidate_lanes)], candidate_lanes):
        score_strings.append("%.2f"%score)
        locations.append(lane[-1])


    # Select with biggest score
    index = np.argmax(scores)

    ref_lane = candidate_lanes[index]

    del candidate_lanes[index]


    image = draw_map(avm, source[-1], city)

    image = draw_2(source, s_line=ref_lane, t_lines=candidate_lanes, other_lines=predictions, image=image)

    image = draw_scores(source[-1], scores=score_strings, locations=locations, image=image)

    # image = cv2.flip(image, 1)

    cv2.imwrite(output_file%(i+1), image)

