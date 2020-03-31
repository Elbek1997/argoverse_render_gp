import pandas as pd

df = pd.DataFrame([ [-1, 0], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [3, 6], [4, 7], [4, 8]] ,columns=["parent_id", "id"])

#region Find first intersection id
intersection_id = -1

while True:
    lines = df.loc[df["parent_id"]==intersection_id]

    if len(lines)>1:
        break
    intersection_id = lines.iloc[0]["id"]

#endregion

print(intersection_id)

print(df)

#region Remove all neighbor lines from target line
id = 7

while True:
    
    lines = df.loc[df["id"]==id]

    if len(lines)<=0:
        break

    parent_id = lines.iloc[0]["parent_id"]

    if parent_id==intersection_id:
        break

    df = df[ (df.parent_id!=parent_id) | (df.id==id) ]

    id = parent_id
#endregion

#region Remove unnecessary lines

for _, row in df.loc[df["parent_id"]==intersection_id].iterrows():
    
    parent_id = row["id"]
    lines = df[ df["parent_id"]==parent_id ]

    while len(lines)>0:
        new_id = lines.iloc[0]["id"]
        df = df[ (df.parent_id!=parent_id) | (df.id==new_id) ]

        parent_id = new_id

        lines = df[ df["parent_id"]==parent_id ]
        




#endregion


print(df)
lines = [ [row["id"]] for _, row in df.loc[df["parent_id"]==-1].iterrows()]

done = False

while not done:
  done = True
  last = [l[-1] for l in lines]

  for i, parent_id in enumerate(last):

    new_ids = [ row["id"] for _, row in df.loc[df["parent_id"]==parent_id].iterrows()]
    
    if len(new_ids)>0:
      done = False
      lines.extend( [lines[i] + [id] for id in new_ids[1::]] )
      lines[i].append(new_ids[0])

print(lines)