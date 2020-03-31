import pickle as pkl
import numpy as np
from tqdm import tqdm
import pandas as pd

from os import listdir
from os.path import isfile, join

root_dir = 'dataset/val'
save_pkl = "val_raw.pkl"

limit = None
# Scan files
files = [ join(root_dir, f) for f in listdir(root_dir) if f.endswith(".csv")][:limit]
length = len(files)

print('Total number of sequences:', length)
recs = []

for i in tqdm(range(length), desc='Loading Argoverse...'):
        # Seq_df
    df = pd.read_csv(files[i])

    # Agent Trajectories 9621829
    agent_x = df[df["OBJECT_TYPE"] == "AGENT"]["X"]
    agent_y = df[df["OBJECT_TYPE"] == "AGENT"]["Y"]
    xy = np.column_stack((agent_x, agent_y))

    city_name = df.iloc[0]["CITY_NAME"] 
    recs.append( {"city": city_name, "traj": xy} )

print("Saving %s"%save_pkl)
pkl.dump(recs, open(save_pkl, "wb"))