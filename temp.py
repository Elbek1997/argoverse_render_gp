def filter(candidate_cl, trajectory):
    # This function is called after lanes are filtered with ROI

    check = lambda traj_start, traj_end, line_start, line_end:
    distance(traj_start, line_start)<=distance(traj_end, line_start) and distance(traj_start, line_end)>=distance(traj_end, line_end)

    return [ line for line in candidate_cl if check(trajectory[0], trajectory[-1], line[0], line[-1])]