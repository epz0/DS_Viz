"""The dspace_metrics module calculates the overall metrics for each participants' exploration of the design space. The focus is on the area covered by participants design space (convex hull).

Functions:
    export_cvxh_metrics: Function that exports the area metrics to an Excel file.
    area_summary: Function to create a dataframe compiling the area metrics for each participant in all stages of the session (Pre, Post and Full Session).
    area_alternative_metrics: Function to create a dataframe compiling the area metrics, including alternative ones (FS-Pre, Overlap, RAE), in all stages of the session (Pre, Post and Full Session).
"""

import numpy as np
import pandas as pd

from pathlib import Path
from shapely import intersection
from shapely import Polygon as Pol
from matplotlib import pyplot as plt

VALID_MODE = {'all', 'full', 'pre', 'post'}

def export_cvxh_metrics(dir_data,  fname, df_ch_fs_metrics,  mode='full', df_ch_pre_metrics=None, df_ch_post_metrics=None):
    """Exports the convex hull metrics (area) to an Excel file.

    Args:
        dir_data (path): Path to the directory with the data file.
        fname (string): _description_
        df_ch_fs_metrics (dataframe): Dataframe with full session convex hull metrics.
        mode (str, optional): The focus of the calculation of the convex hull (all, pre, post, full). Defaults to 'full' for saving only the full session metrics.
        df_ch_pre_metrics (dataframe, optional): Dataframe with pre intervention convex hull metrics. Defaults to None.
        df_ch_post_metrics (dataframe, optional): Dataframe with post intervention convex hull metrics. Defaults to None.

    Raises:
        ValueError: Mode must be one of {all, pre, post, full}.
        ValueError: Incomplete arguments - dfs passed must match the mode selected.
    """

    #* error if mode is invalid
    if mode not in VALID_MODE:
        raise ValueError("mode: mode must be one of %r." % VALID_MODE)

    dir_metrics = Path(f'{dir_data.parent}'+r'/export/metrics')

    print("Exporting convex hull metrics to Excel...")
    # create a excel writer object
    with pd.ExcelWriter(f'{dir_metrics}/{fname}.xlsx') as writer:
        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        match mode:
            case 'all':
                if df_ch_pre_metrics is None or df_ch_post_metrics is None:
                    raise ValueError("Incomplete arguments: mode is {mode} but some of the required dfs are missing.")
                else:
                    df_ch_fs_metrics.to_excel(writer, sheet_name="FullIntervention", index=False)
                    df_ch_pre_metrics.to_excel(writer, sheet_name="Pre", index=False)
                    df_ch_post_metrics.to_excel(writer, sheet_name="Post", index=False)

            case 'full':
                df_ch_fs_metrics.to_excel(writer, sheet_name="FullIntervention", index=False)

            case 'pre':
                if df_ch_pre_metrics is None:
                    raise ValueError("Incomplete arguments: mode is {mode} but the required df is missing.")
                else:
                    df_ch_pre_metrics.to_excel(writer, sheet_name="Pre", index=False)

            case 'post':
                if df_ch_post_metrics is None:
                    raise ValueError("Incomplete arguments: mode is {mode} but the required df is missing.")
                else:
                    df_ch_post_metrics.to_excel(writer, sheet_name="Post", index=False)
    print("Exporting convex hull metrics done!")

def area_summary(df, DS_area, df_ch_fs_metrics, mode='full', df_ch_pre_metrics=None, df_ch_post_metrics=None):
    """Returns dataframe with participants' DS exploration metrics (absolute and percentual).

    Args:
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        DS_area (float): Area of the DS of all solutions (participants & gallery).
        df_ch_fs_metrics (dataframe): Dataframe with full session convex hull metrics.
        mode (str, optional): The focus of the calculation of the convex hull (all, pre, post, full). Defaults to 'full' for saving only the full session metrics.
        df_ch_pre_metrics (dataframe, optional): Dataframe with pre intervention convex hull metrics. Defaults to None.
        df_ch_post_metrics (dataframe, optional): Dataframe with post intervention convex hull metrics. Defaults to None..

    Raises:
        ValueError: Mode must be one of {all, pre, post, full}.
        ValueError: Incomplete arguments - dfs passed must match the mode selected.

    Returns:
        df_DScover: Dataframe with participants' exploration. Must be checked to see if contains the columns for FS/PRE/POST downstream.
    """
    #* error if mode is invalid
    if mode not in VALID_MODE:
        raise ValueError("mode: mode must be one of %r." % VALID_MODE)

    print("Calculating area coverage metrics...")
    #* preparing df: [Group, PID, Area{PRE, POST, FS}, Area Perc{PRE, POST, FS}]
    df_DScover = df[['ParticipantID', 'GroupID']].drop_duplicates().reset_index(drop=True)

    match mode:
            case 'all':
                if df_ch_pre_metrics is None or df_ch_post_metrics is None:
                    raise ValueError("Incomplete arguments: mode is {mode} but some of the required dfs are missing.")
                else:
                    # getting area values from cvx hull calc
                    df_DScover = pd.merge(df_DScover, df_ch_fs_metrics, on="ParticipantID", how="left")
                    df_DScover = pd.merge(df_DScover, df_ch_pre_metrics, on="ParticipantID", how="left")
                    df_DScover =  pd.merge(df_DScover, df_ch_post_metrics, on="ParticipantID", how="left")

                    # tidying the df
                    df_DScover = df_DScover.rename(columns={'CH_Area_x': 'Area-FS', 'CH_Area_y': 'Area-KYY', 'CH_Area': 'Area-ZSB'})        #rename merged columns
                    df_DScover = df_DScover.drop(columns=['CH_Perim_x', 'CH_Perim_y', 'CH_Perim'])                                          # drop perimeter metric
                    df_DScover = df_DScover[['ParticipantID', 'GroupID', 'Area-KYY', 'Area-ZSB', 'Area-FS']].replace(np.nan, 0)             # KYY = pre, ZSB = post
                    print(df_DScover)
                    df_DScover = df_DScover[:-1]                                                                                      # gallery metrics

                    # area covered by each participant in % of whole DS
                    df_DScover['Area-Perc-KYY'] = ((df_DScover['Area-KYY'] *100) / DS_area)
                    df_DScover['Area-Perc-ZSB'] = ((df_DScover['Area-ZSB'] *100) / DS_area)
                    df_DScover['Area-Perc-FS'] = ((df_DScover['Area-FS'] *100) / DS_area)

            case 'full':
                # getting area values from cvx hull calc
                df_DScover = pd.merge(df_DScover, df_ch_fs_metrics, on="ParticipantID", how="left")

                # tidying the df
                df_DScover = df_DScover.rename(columns={'CH_Area': 'Area-FS'})                                                          #rename merged columns
                df_DScover = df_DScover.drop(columns=['CH_Perim'])                                                                      # drop perimeter metric
                df_DScover = df_DScover[['ParticipantID', 'GroupID', 'Area-FS']].replace(np.nan, 0)                                     # KYY = pre, ZSB = post
                df_DScover = df_DScover[:-1]                                                                                        # gallery metrics

                # area covered by each participant in % of whole DS
                df_DScover['Area-Perc-FS'] = ((df_DScover['Area-FS'] *100) / DS_area)

            case 'pre':
                if df_ch_pre_metrics is None:
                    raise ValueError("Incomplete arguments: mode is {mode} but the required df is missing.")
                else:
                    # getting area values from cvx hull calc
                    df_DScover = pd.merge(df_DScover, df_ch_pre_metrics, on="ParticipantID", how="left")

                    # tidying the df
                    df_DScover = df_DScover.rename(columns={'CH_Area': 'Area-KYY'})                                                         #rename merged columns
                    df_DScover = df_DScover.drop(columns=['CH_Perim'])                                                                      # drop perimeter metric
                    df_DScover = df_DScover[['ParticipantID', 'GroupID', 'Area-KYY']].replace(np.nan, 0)                                    # KYY = pre, ZSB = post
                    df_DScover = df_DScover[:-1]                                                                                        # gallery metrics

                    # area covered by each participant in % of whole DS
                    df_DScover['Area-Perc-KYY'] = ((df_DScover['Area-KYY'] *100) / DS_area)

            case 'post':
                if df_ch_post_metrics is None:
                    raise ValueError("Incomplete arguments: mode is {mode} but the required df is missing.")
                else:
                    # getting area values from cvx hull calc
                    df_DScover =  pd.merge(df_DScover, df_ch_post_metrics, on="ParticipantID", how="left")

                    # tidying the df
                    df_DScover = df_DScover.rename(columns={'CH_Area': 'Area-ZSB'})                                                         #rename merged columns
                    df_DScover = df_DScover.drop(columns=['CH_Perim'])                                                                      # drop perimeter metric
                    df_DScover = df_DScover[['ParticipantID', 'GroupID', 'Area-ZSB']].replace(np.nan, 0)                                    # KYY = pre, ZSB = post
                    df_DScover = df_DScover[:-1]                                                                                       # gallery metrics

                    # area covered by each participant in % of whole DS
                    df_DScover['Area-Perc-ZSB'] = ((df_DScover['Area-ZSB'] *100) / DS_area)
    print("Calculating area coverage metrics done!")
    return df_DScover

def area_alternative_metrics(df, pt_cvxh, DS_area, df_DS_vtx, save_plot=False, save_metrics=False, dir_data=None):
    """Returns dataframe with DS exploration metrics (Areas, Areas %, FS-Pre, Overlap, RAE)

    Args:
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        pt_cvxh (dataframe): Dataframe with convex hull metrics for the full session for each participant.
        DS_area (float): Area of the DS of all solutions (participants & gallery).
        df_DS_vtx (dataframe): Dataframe with x,y values of the vertices of the whole DS.
        save_plot (bool, optional): Argument to indicate whether overlap plots should be saved. Defaults to False.
        save_metrics (bool, optional): Argument to indicate whether metrics plots should be saved. Defaults to False.
        dir_data (path, optional): Path to the directory with the data file. Defaults to None, must be passed if save_plot=True.

    Returns:
        pt_cvxh_groupid: Dataframe with alternative participants' DS metrics.
    """

    pt_cvxh['OverlapKYY_ZSB'] = ''
    pt_cvxh['AreaKYY'] = ''
    pt_cvxh['AreaZSB'] = ''

    print("Calculating alternative DS exploration metrics...")

    print("Calculating pre/post overlap...")
    for i in range(len(pt_cvxh['ParticipantID'].unique())):
        plt.close()
        fig, ax1 = plt.subplots()

        ax1.plot(df_DS_vtx['x'], df_DS_vtx['y'], 'o', mec='k', color='none', lw=1, markersize=10)

        #ax1.set_xlim(left=7.5, right=16.1)          #* needs to be update based on the DS limits
        #ax1.set_ylim(bottom=0, top=10)              #* needs to be update based on the DS limits

        polKYY = Pol(pt_cvxh.loc[i,'KYY_CH_Vert'])
        polZSB = Pol(pt_cvxh.loc[i,'ZSB_CH_Vert'])

        if np.size(pt_cvxh.loc[i,'KYY_CH_Vert']) == 1 | np.size(pt_cvxh.loc[i,'ZSB_CH_Vert']) == 1:
            intersec_area = 0
        else:
            intersec_points = intersection(polKYY, polZSB)
            if intersec_points.is_empty == True:
                intersec_area=0
            else:
                intersec_pol = Pol(intersec_points)
                intersec_area = intersec_pol.area

        #pt_cvxh.at[i,'OverlapKYY-ZSB'] = intersec_area
        pt_cvxh.at[i,'AreaKYY'] = polKYY.area
        pt_cvxh.at[i,'AreaZSB'] = polZSB.area
        pt_cvxh.at[i,'OverlapKYY_ZSB'] = intersec_area

        if save_plot == True:
            if dir_data  is None:
                raise ValueError(f"save_plot = {save_plot} requires a valid dir_data for saving the files")
            else:
                dir_viz = Path(f'{dir_data.parent}'+r'/viz/DS_cvxh')

                x,y = polKYY.exterior.xy
                plt.plot(x,y)

                x,y = polZSB.exterior.xy
                plt.plot(x,y)

                if intersec_area != 0:
                    x,y = intersec_pol.exterior.xy
                    plt.plot(x,y)

                plt.title(f'DS-Viz {pt_cvxh.loc[i, "ParticipantID"]} | Overlap Pre/Post in Green', fontsize=16, fontweight='bold')
                #plt.show()
                plt.savefig(f'{dir_viz}/DS_{pt_cvxh.loc[i, "ParticipantID"]}_Overlap', dpi=300, bbox_inches='tight')



    print(f"Calculating pre/post overlap done! save_plot={save_plot}")
    #* Preparing df for metrics calc
    #* Extra Exploration [FS - PRE]
    pt_cvxh['ExtraExp'] = pt_cvxh['AreaFS'] - pt_cvxh['AreaKYY']

    #* Remaining Area Explored - RAE%
    #* (Post - Overlap)/(DS - Pre)
    pt_cvxh['RAE_perc'] = (pt_cvxh['AreaZSB'] - pt_cvxh['OverlapKYY_ZSB'])*100/(DS_area - pt_cvxh['AreaKYY'])
    pt_cvxh['AreaKYY_perc'] = (pt_cvxh['AreaKYY']*100/DS_area)

    #* Tidying up DFs
    df_groupid = df[['ParticipantID', 'GroupID']].drop_duplicates().reset_index(drop=True)
    pt_cvxh_groupid = pd.merge(left=pt_cvxh, right=df_groupid, on='ParticipantID', how='left')
    pt_cvxh_groupid = pt_cvxh_groupid[:-1]

    pt_cvxh_groupid = pt_cvxh_groupid.drop(columns=['KYY_CH_Vert', 'ZSB_CH_Vert', 'FS_CH_Vert'])
    pt_cvxh_groupid['AreaKYY'] = pd.to_numeric(pt_cvxh_groupid['AreaKYY'])
    pt_cvxh_groupid['AreaKYY_perc'] = pd.to_numeric(pt_cvxh_groupid['AreaKYY_perc'])
    pt_cvxh_groupid['AreaZSB'] = pd.to_numeric(pt_cvxh_groupid['AreaZSB'])
    pt_cvxh_groupid['ExtraExp'] = pd.to_numeric(pt_cvxh_groupid['ExtraExp'])
    pt_cvxh_groupid['RAE_perc'] = pd.to_numeric(pt_cvxh_groupid['RAE_perc'])

    if save_metrics == True:
        if dir_data is None:
            raise ValueError(f"save_metrics = {save_metrics} requires a valid dir_data for saving the files")

        else:
            dir_metrics = Path(f'{dir_data.parent}'+r'/export/metrics')
            with pd.ExcelWriter(f'{dir_metrics}/DSX_Alt_Metrics.xlsx') as writer:
                pt_cvxh_groupid.to_excel(writer, sheet_name="AllMetrics", index=False)

    print(f"Calculating alternative DS exploration metrics done! save_metrics={save_metrics}")
    return pt_cvxh_groupid
