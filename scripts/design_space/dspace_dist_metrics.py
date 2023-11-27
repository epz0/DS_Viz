"""The dspace_dist_metrics module calculates the distance metrics for each participants' exploration of the design space. This includes the average, maximum and total lenght of the participants' path.

Functions:
    dist_metrics_fs: Main function that creates the metrics dataframe and calculate the distance metrics for each participants' full session.
    dist_metrics_pre: Function to create distance metrics for each participants' pre intervention part of the session.
    dist_metrics_post: Function to create distance metrics for each participants' pre intervention part of the session.
"""

import numpy as np

def dist_metrics_fs(df_unmask, dist_matrix):
    """Returns a dataframe with the Full Session distance metrics (max, avg, total) for each participant and list of participants.

    Args:
        df_unmask (dataframe): Dataframe with all main data, and with the unmasking applied to it.
        dist_matrix (matrix):  Normalised matrix with pairwise distances.

    Returns:
        df_dist: Dataframe populated with the distance metrics for the Full Session only.
        pt_unique: List of unique participant IDs.
    """

    print("Calculating distance metrics for the full session...")
    print(df_unmask)
    col_ID_PT = 'ParticipantID'                 #! column with the participants' ids
    col_ID_GP = 'GroupID'                       #! column with the group ids
    col_ID_SOL = 'SolutionID'



    # list of unique participants from unmasked data
    pt_unique = df_unmask[f'{col_ID_PT}'].unique()
    pt_unique = pt_unique[:-1]                      #drop the gallery

    # output df_dist
    df_dist = df_unmask[[f'{col_ID_PT}', f'{col_ID_GP}']].drop_duplicates().reset_index(drop=True)
    df_dist = df_dist[:-1]

    # initialising columns for the max dist
    df_dist['maxdist_FS'] = 0
    df_dist['maxdist_PRE'] = 0
    df_dist['maxdist_PST'] = 0

    # initialising columns for the total dist
    df_dist['totaldist_FS'] = 0
    df_dist['totaldist_PRE'] = 0
    df_dist['totaldist_PST'] = 0
    df_dist['dist_PRE_POST'] = 0

    # initialising columns for the total dist
    df_dist['avgdist_FS'] = 0
    df_dist['avgdist_PRE'] = 0
    df_dist['avgdist_PST'] = 0

    #* initialising column for the ordered path indices
    df_dist['ord_idx_FS'] = ''
    df_dist['ord_idx_PRE'] = ''
    df_dist['ord_idx_PST'] = ''

    # get metrics dist for each participant - #!FULL SESSION
    for i in range(len(pt_unique)):

        rows_PT = df_unmask[df_unmask[f'{col_ID_PT}'] == pt_unique[i]] # get the lines corresponding to each participant

        # get list of solutions ids
        n_sol_PT = len(rows_PT)+1               # to account for the base zero py
        solutions_list = list(range(n_sol_PT))  # getting the list
        solutions_list.pop(0)                   # dropping the zero

        # list of solution index, based on the solution id
        solution_idx = []
        for j in solutions_list:
            solution_idx.append(rows_PT.index[rows_PT[f'{col_ID_SOL}'] == j].tolist()[0])

        # solutions_list --> ordered solutions (1 --> N, size N)
        # solution_idx --> the corresponding index of the solution number in te matrix (size N)
        # path_dist --> the distance value between solution_idx[i, i+1] (size N-1)

        # getting the distance between the solutions from the high dim matrix
        # and calculating total dist, maxdist, avgdist

        path_dist = []

        for s_idx in range(len(solution_idx)-1):
            path_dist.append(dist_matrix[solution_idx[s_idx]][solution_idx[s_idx+1]])

        path_solutions = np.array(path_dist)

        total_dist = np.sum(path_solutions)
        max_dist = np.max(path_solutions)
        mean_dist = np.mean(path_solutions)

        df_dist.at[i, 'totaldist_FS'] = total_dist
        df_dist.at[i, 'maxdist_FS'] = max_dist
        df_dist.at[i, 'avgdist_FS'] = mean_dist
        df_dist.at[i, 'ord_idx_FS'] = solution_idx
    print("Calculating distance metrics for the full session done!")
    return df_dist, pt_unique

def dist_metrics_pre(df_unmask, df_dist, pt_unique, dist_matrix):
    """Returns a dataframe with the distance metrics (max, avg, total), including Pre intervention metrics, for each participant.

    Args:
        df_unmask (dataframe): Dataframe with all main data, and with the unmasking applied to it.
        df_dist (dataframe): Dataframe populated with the distance metrics.
        pt_unique: List of unique participant IDs.
        dist_matrix (matrix):  Normalised matrix with pairwise distances.

    Returns:
        df_dist: Dataframe populated with the distance metrics, including Pre intervention metrics.
    """

    col_ID_PT = 'ParticipantID'                 #! column with the participants' ids
    col_ID_PrePost = 'PrePost'                  #! column with the prepost ids
    col_ID_SOL = 'SolutionID'



    print("Calculating distance metrics for the pre intervention part of the session...")
    # get metrics dist for each participant - #!PRE
    #* subset the df_base to have only PRE solutions
    df_PRE = df_unmask[df_unmask[f'{col_ID_PrePost}'].eq("Pre")]

    for i in range(len(pt_unique)):
        rows_PT = df_PRE[df_PRE[f'{col_ID_PT}'] == pt_unique[i]] # get the lines corresponding to each participant

        # get list of solutions ids
        n_sol_PT = len(rows_PT)+1               # to account for the base zero py
        solutions_list = list(range(n_sol_PT))  # getting the list
        solutions_list.pop(0)                   # dropping the zero

        # list of solution index, based on the solution id
        solution_idx = []
        for j in solutions_list:
            solution_idx.append(rows_PT.index[rows_PT[f'{col_ID_SOL}'] == j].tolist()[0])

        # solutions_list --> ordered solutions (1 --> N, size N)
        # solution_idx --> the corresponding index of the solution number in te matrix (size N)
        # path_dist --> the distance value between solution_idx[i, i+1] (size N-1)

        # getting the distance between the solutions from the high dim matrix
        # and calculating total dist, maxdist, avgdist

        path_dist = []

        for s_idx in range(len(solution_idx)-1):
            path_dist.append(dist_matrix[solution_idx[s_idx]][solution_idx[s_idx+1]])

        path_solutions = np.array(path_dist)

        total_dist = np.sum(path_solutions)
        max_dist = np.max(path_solutions)
        mean_dist = np.mean(path_solutions)

        df_dist.at[i, 'totaldist_PRE'] = total_dist
        df_dist.at[i, 'maxdist_PRE'] = max_dist
        df_dist.at[i, 'avgdist_PRE'] = mean_dist
        df_dist.at[i, 'ord_idx_PRE'] = solution_idx

    print("Calculating distance metrics for the pre intervention part of the session done!")
    return df_dist

def dist_metrics_post(df_unmask, df_dist, pt_unique, dist_matrix):
    """Returns a dataframe with the distance metrics (max, avg, total), including Post intervention metrics, for each participant.

    Args:
        df_unmask (dataframe): Dataframe with all main data, and with the unmasking applied to it.
        df_dist (dataframe): Dataframe populated with the distance metrics.
        pt_unique: List of unique participant IDs.
        dist_matrix (matrix):  Normalised matrix with pairwise distances.

    Returns:
        df_dist: Dataframe populated with the distance metrics, including Post intervention metrics.
    """

    col_ID_PT = 'ParticipantID'                 #! column with the participants' ids
    col_ID_PrePost = 'PrePost'                  #! column with the prepost ids
    col_ID_SOL = 'SolutionID'

    print("Calculating distance metrics for the post intervention part of the session...")
    # get metrics dist for each participant - #!POST
    # post - a bit different -nsol has to start from when post started

    #* subset the df_base to have only POST solutions
    df_PST = df_unmask[df_unmask[f'{col_ID_PrePost}'].eq("Pst")]

    for i in range(len(pt_unique)):
        # get list with all solutions for the participant
        nsol_PT_all = len(df_unmask[df_unmask[f'{col_ID_PT}'] == pt_unique[i]])+1
        solutions_list_all = list(range(nsol_PT_all))  # getting the list
        solutions_list_all.pop(0)

        # get list with pre solutions
        nsol_PT_pre = len(df_unmask[(df_unmask[f'{col_ID_PT}'] == pt_unique[i]) & (df_unmask[f'{col_ID_PrePost}'] == 'Pre')])+1
        solutions_list_pre = list(range(nsol_PT_pre))  # getting the list
        solutions_list_pre.pop(0)

        rows_PT = df_PST[df_PST[f'{col_ID_PT}'] == pt_unique[i]] # get the lines corresponding to each participant

        #* get list of solutions post
        solutions_list = list(set(solutions_list_all).difference(solutions_list_pre))

        # list of solution index, based on the solution id
        solution_idx = []
        for j in solutions_list:
            solution_idx.append(rows_PT.index[rows_PT[f'{col_ID_SOL}'] == j].tolist()[0])

        # solutions_list --> ordered solutions (1 --> N, size N)
        # solution_idx --> the corresponding index of the solution number in te matrix (size N)
        # path_dist --> the distance value between solution_idx[i, i+1] (size N-1)

        # getting the distance between the solutions from the high dim matrix
        # and calculating total dist, maxdist, avgdist

        path_dist = []

        for s_idx in range(len(solution_idx)-1):
            path_dist.append(dist_matrix[solution_idx[s_idx]][solution_idx[s_idx+1]])

        path_solutions = np.array(path_dist)

        if len(solution_idx) < 2:
            total_dist = 0
            max_dist = 0
            mean_dist = 0
        else:
            total_dist = np.sum(path_solutions)
            max_dist = np.max(path_solutions)
            mean_dist = np.mean(path_solutions)

        df_dist.at[i, 'totaldist_PST'] = total_dist
        df_dist.at[i, 'maxdist_PST'] = max_dist
        df_dist.at[i, 'avgdist_PST'] = mean_dist
        df_dist.at[i, 'ord_idx_PST'] = solution_idx

        distprepost = dist_matrix[df_dist.loc[i,'ord_idx_PRE'][-1]][df_dist.loc[i,'ord_idx_PST'][0]]
        df_dist.at[i, 'dist_PRE_POST'] = distprepost

    print("Calculating distance metrics for the post intervention part of the session done!")
    return df_dist