from utils import *
from tqdm import tqdm
import cv2

from shapely.geometry import Polygon

from argoverse.map_representation.map_api import ArgoverseMap

data = load_pickle("input/val.pkl")

output = "test/val_candidates/%d.jpg"

avm = ArgoverseMap()


limit = len(data)

diff = 0


for i in tqdm(range(limit), desc="Processing"):


    # rec = loader.get(root_dir + "/%d.csv"%(ids[i]))
    # rec = data[indices[i]]
    rec = data[i]

    # City
    city = rec["city"]

    source = rec["source_trj"]
    target = rec["target_trj"]
    last_point = Point(target[-1])

    
    other_lines, roi = get_candidate_centerlines(avm, source, city)

    label = closest_line(other_lines, target)

    label_line = other_lines[label]

    if rec["label"]!=label:
        diff = diff + 1

        image = draw_map(avm, source[-1], city)

        image = draw(source, image=image, other_lines=other_lines, polygons=[roi]
            , label_line=label_line, s_line=other_lines[rec["label"]], target=target
            )

        image = cv2.flip(image, 1)

        # seq_id = indices[i]

        # cv2.imwrite(output%(seq_id), image)

        cv2.imwrite(output%(i+1), image)

        


# print("Difference %d"%diff)

# print(indices)



