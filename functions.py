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

def combine(conn):
    lines = []

    for i, row in conn.iterrows():
        id = row["LANE_ID"]
        line = [id]
        # Forward
        forward_id = id
        while True:

            sel = conn[conn["LANE_ID"]==forward_id]
            if len(sel)==0 or sel.iloc[0]["OUT_LANE"] is None:
                break

            forward_id = sel.iloc[0]["OUT_LANE"]
            if len(df[ df["LANE_ID"]==forward_id ])==0:
                break
            line.append(forward_id)
        
        # # Backward
        # backward_id = id
        # while True:

        #     sel = conn[conn["LANE_ID"]==backward_id]
        #     if len(sel)==0 or sel.iloc[0]["IN_LANE"] is None:
        #         break

        #     backward_id = sel.iloc[0]["IN_LANE"]
        #     if len(df[df["LANE_ID"]==backward_id])==0:
        #         break
        #     line = [backward_id] + line

        lines.append(line)

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
    return final
            

def merge(lines):
    final = []

    for line in lines:
        final = final + list(line.coords)
    
    return LineString(final)

def build_line(df, ids):
    lines = [ df[ df["LANE_ID"]==id ].iloc[0]["geometry"] for id in ids]

    return merge(lines)