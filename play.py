from utils import *
from tqdm import tqdm
import cv2

from shapely.geometry import Polygon

from argoverse.map_representation.map_api import ArgoverseMap

root_dir = "dataset/train"
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
loader = ArgoverseForecastingLoader(root_dir)

output = "test/train_candidates/%d.jpg"

avm = ArgoverseMap()

limit = 100

# ids = [80090, 57719, 5929, 23905, 14429, 76983, 79514, 2919, 77819, 42613, 71895, 48768, 35032, 64539, 23994, 64714, 44601, 1321, 72561, 31035, 18906, 36271, 54210, 13534, 59308, 49241, 30375, 46025, 53448, 23334, 36983, 56157, 65343, 41714, 65024, 21393, 14162, 29513, 80346, 74519, 11271, 27379, 66448, 5007, 36466, 49308, 57987, 72210, 54290, 31029, 38642, 66260, 61778, 68773, 66588, 44472, 21794, 36541, 23772, 73796, 38815, 71918, 58768, 43290, 28556, 6103, 78493, 19997, 16493, 46354, 68548, 4964, 64535, 3442, 37666, 29576, 1793, 72580, 46838, 52382, 11759, 59271, 43917, 9134, 44713, 80577, 80175, 4351, 29658, 72039, 5276, 4831, 77564, 77097, 33429, 15470, 41813, 49744, 24113, 65319, 3839, 40063, 18719, 73396, 21065, 59191, 45637, 54571, 51907, 54011, 10608, 74133, 29029, 19137, 41299, 26157, 9263, 8623, 2304, 66051, 20295, 54031, 25869, 64295, 47345, 77761, 1248, 63418, 39047, 68668, 65544, 27440, 40195, 28583, 57172, 30016, 45086, 25164, 15486, 51935, 77491, 2767, 7857, 27242, 74470, 62490, 55577, 19186, 23722, 38543, 40808, 30277, 12818, 7533, 6741, 40432, 10536, 36627, 47754, 26612, 50347, 1028, 79897, 66281, 18652, 72211, 23004, 62993, 28678, 8184, 52042, 52305, 38989, 49676, 15385, 65054, 18328, 19215, 64824, 47195, 6091, 40005, 13991, 77575, 35796, 50604, 36580, 23910, 11816, 43101, 56882, 66844, 374, 14786, 1924, 39023, 42554, 17182, 76391, 25464, 58546, 56911, 30722, 58526, 16295, 8632, 61084, 58671, 4569, 63295, 41529, 43987, 59931, 32200, 63314, 8192, 80016, 46326, 66595, 53649, 66323, 13646, 21672, 7116, 8055, 74469, 76726, 76884, 1049, 26579, 2655, 78781, 44115, 38957, 6954, 76948, 52583, 46402, 22368, 28060, 43412, 69852, 63102, 68078, 49976, 52613, 16437, 50948, 44686, 36990, 31581, 24302, 10276, 25977, 54020, 39535, 25937, 67791, 79279, 79434, 75390, 72414, 55030, 22634, 47560, 74899, 61715, 57323, 49449, 37564, 12375, 79620, 63627, 14409, 53086, 63798, 37148, 72107, 62898, 13594, 71889, 73116, 58023, 39057, 15818, 9764, 58019, 66049, 46363, 21451, 76101, 36618, 71382, 78052, 36316, 68147, 68074, 1031, 77546, 80011, 12850, 22574, 33302, 34379, 34140, 78705, 62118, 3324, 42155, 72281, 18136, 53595, 74397, 14, 20635, 40762, 2756, 20816, 43174, 22613, 915, 49816]

ids = [6018]

indices = list(range(len(loader)))

np.random.shuffle(indices)

indices = indices[:limit]


for i in tqdm(range(limit), desc="Processing"):


    # rec = loader.get(root_dir + "/%d.csv"%(ids[i]))
    rec = loader[indices[i]]

    # City
    city = rec.city

    traj = rec.agent_traj

    source = traj[:20]
    target = traj[20:]
    last_point = Point(target[-1])

    
    other_lines, roi = get_candidate_centerlines(avm, source, city)

    label = sorted(range(len(other_lines)), key=lambda k: (hausdorffDistance(target, other_lines[k]), hausdorffDistance(traj, other_lines[k]) , LineString(other_lines[k]).distance(last_point)) )[0]

    label_line = other_lines[label]
    
    del other_lines[label]

    

    image = draw_map(avm, source[-1], city)

    image = draw(source, image=image, other_lines=other_lines, polygons=[roi]
        , label_line=label_line, target=target
        )

    image = cv2.flip(image, 1)

    seq_id = int(rec.current_seq.name[:-4])

    cv2.imwrite(output%(seq_id), image)

    # if len(other_lines)<=0:
    #     print(i+1)


    





