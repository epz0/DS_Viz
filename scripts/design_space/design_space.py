"""The design space module creates the plots and metrics of the participants' design space.

Functions:
    create_cvxh: Main function that controls which plots/metrics will be generated.
    plot_full: Function to create plots/metrics for each participant, in the full session.
    plot_pre: Function to create plots/metrics for each participant, for the pre intervention part of the session.
    plot_post: Function to create plots/metrics for each participant, for the post intervention part of the session.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd

from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from shapely import intersection
from shapely import Polygon as Pol
from shapely.geometry import Point

VALID_MODE = {'all', 'full', 'pre', 'post'}


def plot_full(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot=False, mult_plot=None):
    """Calculate and plot the design space for the full session for each participant.

    Args:
        dir_data (path): Path to the directory with the data file.
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        pt_cvxh (dataframe): Dataframe that will receive convex hull metrics for each participant.
        embed (matrix): Full embedding with x,y points coordinates.
        df_colors (dataframe): Dataframe with color scheme for the participants.
        df_DS_vertx (dataframe): Empty dataframe for the vertices of the whole DS.
        save_plot (bool, optional): Parameter that defines if plots will be created and saved. Defaults to False.
        mult_plot (string, optional): Parameter for multi plot naming when validating. Defaults to None.

    Returns:
        DS_area (float): Area of the DS of all solutions (participants & gallery).
        pt_cvxh (dataframe): Dataframe with convex hull metrics for the full session for each participant.
        df_cvxh_metrics (dataframe): Dataframe with convex hull metrics for the full session for each participant.
        df_DS_vertx (dataframe): Populated dataframe for the vertices of the whole DS.
    """
    dir_viz = Path(f'{dir_data.parent}'+r'/viz/DS_cvxh')

    print(f"Calculating participants' convex hull for the FULL SESSION, save = {save_plot}...")
    # get list of unique participant ids
    pt_unique = df['ParticipantID'].unique()

    # create df to dump convex hull metrics
    df_cvxh_metrics = pd.DataFrame(df['ParticipantID'].unique(), columns=['ParticipantID'])
    df_cvxh_metrics['CH_Area'] = 0
    df_cvxh_metrics['CH_Perim'] = 0
    pt_cvxh['AreaFS'] = 0                                       # adding area Full Session column

    #! loop to create participants' cvxh for the FULL SESSION and save all in the same image
    for i in range(len(pt_unique)):
        rowsPT = df[df['ParticipantID'] == pt_unique[i]]        # get the lines corresponding to each participant
        rowsPT = rowsPT.index.tolist()
        startPT = rowsPT[0]
        endPT = rowsPT[-1]

    # get participants colors
        part_c = df['colors']

    # getting participant corresponding points from UMAP embedding
        all_x_points = embed[:,0]
        all_y_points = embed[:,1]

        x_points = all_x_points[startPT:endPT+1]
        y_points = all_y_points[startPT:endPT+1]

        x_y_points = np.array(list(zip(x_points,y_points)))

        '''# add labels to the points
        labelPT = labels[rowsPT]                                            #getting the labels for the current PT
        for label, x, y in zip(labelPT, x_points, y_points):
            plt.annotate(label, (x,y), xycoords = 'data')'''

    # create convexhull for the participant
        hullPT = ConvexHull(x_y_points)
        pts = x_y_points[hullPT.vertices]

        if save_plot == True:
        #* plotting in the loop to gen each P image
        # set figure size & basic params
            plt.rcParams["figure.figsize"] = (15,15)
            fig, ax = plt.subplots()

        # add points as scatter plot, from x,y of the UMAP embedding
            scatter = ax.scatter(embed[:, 0],embed[:, 1], c=part_c, s=10)

        # get centre & vertices from convex hull
            cent = np.mean(x_y_points, axis=0)
            pts = x_y_points[hullPT.vertices]

            for simplex in hullPT.simplices:
                plt.plot(x_y_points[simplex, 0], x_y_points[simplex, 1], df_colors.loc[i,'HEX-AreaCH']) #transparent markers, with solid edge
            plt.plot(x_y_points[hullPT.vertices, 0], x_y_points[hullPT.vertices, 1], 'o', mec=df_colors.loc[i,'HEX-Win'], color='none', lw=3, markersize=12)

            k = 1
            color = df_colors.loc[i,'HEX-AreaCH']
            poly = Polygon(k*(pts - cent) + cent, closed=True,
                        capstyle='round', facecolor=color, alpha=0.4)
            plt.gca().add_patch(poly)

    # get metrics from convex hull created
        df_cvxh_metrics.at[i, 'CH_Area'] = hullPT.volume
        df_cvxh_metrics.at[i, 'CH_Perim'] = hullPT.area
        pt_cvxh.at[i, 'FS_CH_Vert'] = pts
        pt_cvxh.at[i, 'AreaFS'] = hullPT.volume

        #! FULL DESIGN SPACE CONVEX HULL
        # create convexhull for the full UMAP set of points
        hullDS = ConvexHull(embed)

        # get metrics from convex hull created
        DS_area = hullDS.volume
        DS_perim = hullDS.area

        if save_plot == True:
            # get centre & vertices from convex hull
            cent = np.mean(embed, axis=0)
            pts = embed[hullDS.vertices]

            for simplex in hullDS.simplices:
                plt.plot(embed[simplex, 0], embed[simplex, 1], 'k') # edge
            plt.plot(embed[hullDS.vertices, 0], embed[hullDS.vertices, 1], 'o', mec='k', color='none', lw=1, markersize=10) #transparent markers, with solid edge

            k = 1
            color = 'k'
            poly = Polygon(k*(pts - cent) + cent, closed=True,
                        capstyle='round', facecolor=color, alpha=0.005)
            plt.gca().add_patch(poly)

            if mult_plot is not None:
                # plot annotations
                plt.title(f'DS-Viz ({pt_unique[i]}, Full Session, all results, {mult_plot}', fontsize=16, fontweight='bold')
            else:
                plt.title(f'DS-Viz ({pt_unique[i]}, Full Session, all results', fontsize=16, fontweight='bold')
            #plt.show()

        if save_plot == True:
            if mult_plot is not None:
                plt.savefig(f'{dir_viz}/DS_{pt_unique[i]}_{mult_plot}_FullSession.png', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.savefig(f'{dir_viz}/DS_{pt_unique[i]}_FullSession.png', dpi=300, bbox_inches='tight')
                plt.close()

    df_DS_vertx['x'] = embed[hullDS.vertices, 0]
    df_DS_vertx['y'] = embed[hullDS.vertices, 1]
    return DS_area, pt_cvxh, df_cvxh_metrics, df_DS_vertx

def plot_pre(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot=False, mult_plot=None):
    """Calculate and plot the design space for the pre intervention part of the session for each participant.

    Args:
        dir_data (path): Path to the directory with the data file.
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        pt_cvxh (dataframe): Dataframe that will receive convex hull metrics for each participant.
        embed (matrix): Full embedding with x,y points coordinates.
        df_colors (dataframe): Dataframe with color scheme for the participants.
        df_DS_vertx (dataframe): Empty dataframe for the vertices of the whole DS.
        save_plot (bool, optional): Parameter that defines if plots will be created and saved. Defaults to False.
        mult_plot (string, optional): Parameter for multi plot naming when validating. Defaults to None.

    Returns:
        DS_area (float): Area of the DS of all solutions (participants & gallery).
        pt_cvxh (dataframe): Dataframe with convex hull metrics for the pre intervention part of the session for each participant.
        df_cvxh_metrics_KYY (dataframe): Dataframe with convex hull metrics for the pre intervention part of the session for each participant.
        df_DS_vertx (dataframe): Populated dataframe for the vertices of the whole DS.
    """

    dir_viz = Path(f'{dir_data.parent}'+r'/viz/DS_cvxh')
    print(f"Calculating participants' convex hull for the PRE INTERVENTION part of the session, save = {save_plot}...")

    # df with all participant solutions that happend in the PRE moment
    pre_tag = 'Pre'                                                                                     #* string that tags pre intervention solutions (KYY in this case from the masking)
    ctx_tag = 'UNK'                                                                                     #* string that tag context solutions (UNK from the gallery solutions)

    # preparing data to dfs
    KYY_UKN_lst = df.index[df['PrePost'].eq(pre_tag) | df['PrePost'].eq(ctx_tag)].tolist()              # list of the indices that match the criteria
    df_KYY = df.iloc[KYY_UKN_lst]                                                                       # df with the selected criteria
    id_counts = df_KYY['ParticipantID'].value_counts()                                                  #* to get only the indices of participants
    selected_ids = id_counts[id_counts >= 3].index                                                      #* who had n >=3 at the session step
    df_KYY = df_KYY[df_KYY['ParticipantID'].isin(selected_ids)]                                         #* (necessary for the convex hull
    c_KYY = df_colors[df_colors['P'].isin(selected_ids)].reset_index(drop=True)                         #* same thing for the color df

    # create df to dump convex hull metrics
    pt_unique = df_KYY['ParticipantID'].unique()
    df_cvxh_metrics_KYY = pd.DataFrame(df_KYY['ParticipantID'].unique(), columns=['ParticipantID'])
    df_cvxh_metrics_KYY['CH_Area'] = 0
    df_cvxh_metrics_KYY['CH_Perim'] = 0

    for i in range(len(pt_unique)):
        rowsPT = df_KYY[df_KYY['ParticipantID'] == pt_unique[i]] # get the lines corresponding to each participant, FROM THE DF_KYY
        rowsPT_idx = rowsPT.index.tolist()

        # get participants colors
        part_c = df[df['PrePost'].eq(pre_tag) | df['PrePost'].eq(ctx_tag)]
        part_c = part_c['colors']

    # getting participant corresponding points from UMAP embedding
        all_x_points = embed[:,0]
        all_y_points = embed[:,1]

        x_points = all_x_points[rowsPT_idx]
        y_points = all_y_points[rowsPT_idx]

        x_y_points = np.array(list(zip(x_points,y_points)))

    # create convexhull for the participant
        hullPT = ConvexHull(x_y_points)
        pts = x_y_points[hullPT.vertices]

        if save_plot == True:
            # set figure size & basic params
            plt.rcParams["figure.figsize"] = (15,15)
            fig, ax = plt.subplots()

            # add points as scatter plot, from x,y of the UMAP embedding ONLY FOR THE KYY
            scatter = ax.scatter(embed[KYY_UKN_lst, 0],embed[KYY_UKN_lst, 1], c=part_c, s=10)

        # get centre & vertices from convex hull
            cent = np.mean(x_y_points, axis=0)
            pts = x_y_points[hullPT.vertices]

            for simplex in hullPT.simplices:
                plt.plot(x_y_points[simplex, 0], x_y_points[simplex, 1], c_KYY.loc[i,'HEX-AreaCH']) # transparent markers, with solid edge
            plt.plot(x_y_points[hullPT.vertices, 0], x_y_points[hullPT.vertices, 1], 'o', mec=c_KYY.loc[i,'HEX-Win'], color='none', lw=3, markersize=12)

            k = 1
            color = c_KYY.loc[i,'HEX-AreaCH']
            poly = Polygon(k*(pts - cent) + cent, closed=True,
                        capstyle='round', facecolor=color, alpha=0.4)
            plt.gca().add_patch(poly)

    # get metrics from convex hull created
        df_cvxh_metrics_KYY.at[int(df_cvxh_metrics_KYY[df_cvxh_metrics_KYY['ParticipantID']== pt_unique[i]].index.values), 'CH_Area'] = hullPT.volume   #! placing it in the corresponding line
        df_cvxh_metrics_KYY.at[int(df_cvxh_metrics_KYY[df_cvxh_metrics_KYY['ParticipantID']== pt_unique[i]].index.values), 'CH_Perim'] = hullPT.area    #! getting the index on the df of the
        pt_cvxh.at[int(pt_cvxh[pt_cvxh['ParticipantID'] == pt_unique[i]].index.values), 'KYY_CH_Vert'] = pts                                            #! current id in pt_unique

        #! FULL DESIGN SPACE CONVEX HULL
        # create convexhull for the full UMAP set of points
        hullDS = ConvexHull(embed)

        # get metrics from convex hull created
        DS_area = hullDS.volume
        DS_perim = hullDS.area

        if save_plot == True:
            # get centre & vertices from convex hull
            cent = np.mean(embed, axis=0)
            pts = embed[hullDS.vertices]

            for simplex in hullDS.simplices:
                plt.plot(embed[simplex, 0], embed[simplex, 1], 'k') # edge

            plt.plot(embed[hullDS.vertices, 0], embed[hullDS.vertices, 1], 'o', mec='k', color='none', lw=1, markersize=10) #transparent markers, with solid edge

            k = 1
            color = 'k'
            poly = Polygon(k*(pts - cent) + cent, closed=True,
                        capstyle='round', facecolor=color, alpha=0.005)
            plt.gca().add_patch(poly)

            '''# add labels to the points
                for label, x, y in zip(labels, embedding_umap[:,0], embedding_umap[:,1]):
                    plt.annotate(label, (x,y), xycoords = 'data')'''

            if mult_plot is not None:
                # plot annotations
                plt.title(f'DS-Viz ({pt_unique[i]} Pre, all results, {mult_plot})', fontsize=16, fontweight='bold')
            else:
                plt.title(f'DS-Viz ({pt_unique[i]} Pre, all results)', fontsize=16, fontweight='bold')
            #plt.show()

        if save_plot == True:
            if mult_plot is not None:
                plt.savefig(f'{dir_viz}/DS_{pt_unique[i]}_{mult_plot}_Pre.png', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.savefig(f'{dir_viz}/DS_{pt_unique[i]}_Pre.png', dpi=300, bbox_inches='tight')
                plt.close()
    df_DS_vertx['x'] = embed[hullDS.vertices, 0]
    df_DS_vertx['y'] = embed[hullDS.vertices, 1]

    return DS_area, pt_cvxh, df_cvxh_metrics_KYY, df_DS_vertx

def plot_post(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot=False, mult_plot=None):
    """Calculate and plot the design space for the post intervention part of the session for each participant.

    Args:
        dir_data (path): Path to the directory with the data file.
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        pt_cvxh (dataframe): Dataframe that will receive convex hull metrics for each participant.
        embed (matrix): Full embedding with x,y points coordinates.
        df_colors (dataframe): Dataframe with color scheme for the participants.
        df_DS_vertx (dataframe): Empty dataframe for the vertices of the whole DS.
        save_plot (bool, optional): Parameter that defines if plots will be created and saved. Defaults to False.
        mult_plot (string, optional): Parameter for multi plot naming when validating. Defaults to None.

    Returns:
        DS_area (float): Area of the DS of all solutions (participants & gallery).
        pt_cvxh (dataframe): Dataframe with convex hull metrics for the pre intervention part of the session for each participant.
        df_cvxh_metrics_ZSB (dataframe): Dataframe with convex hull metrics for the pre intervention part of the session for each participant.
        df_DS_vertx (dataframe): Populated dataframe for the vertices of the whole DS.
    """
    dir_viz = Path(f'{dir_data.parent}'+r'/viz/DS_cvxh')
    print(f"Calculating participants' convex hull for the POST INTERVENTION part of the session, save = {save_plot}...")

    # df with all participant solutions that happend in the POST moment
    post_tag = 'Pst'                                                                                    #* string that tags pre intervention solutions (KYY in this case from the masking)
    ctx_tag = 'UNK'                                                                                     #* string that tag context solutions (UNK from the gallery solutions)

    #! loop to create participants' cvxh for the POST INTERVENTION part and save all in the same image
    # get participants colors
    part_c = df[df['PrePost'].eq(post_tag) | df['PrePost'].eq(ctx_tag)]
    part_c = part_c['colors']

    # preparing data to dfs
    ZSB_UKN_lst = df.index[df['PrePost'].eq(post_tag) | df['PrePost'].eq(ctx_tag)].tolist()   #list of the indices that match the criteria
    df_ZSB = df.iloc[ZSB_UKN_lst]
    id_counts = df_ZSB['ParticipantID'].value_counts()                                                  #* to get only the indices of participants
    selected_ids = id_counts[id_counts >= 3].index                                                      #* who had n >=3 at the session step
    df_ZSB = df_ZSB[df_ZSB['ParticipantID'].isin(selected_ids)]                                         #* (necessary for the convex hull
    c_ZSB = df_colors[df_colors['P'].isin(selected_ids)].reset_index(drop=True)                         #*same thing for the color df

    # create df to dump convex hull metrics
    pt_unique = df_ZSB['ParticipantID'].unique()
    df_cvxh_metrics_ZSB = pd.DataFrame(df_ZSB['ParticipantID'].unique(), columns=['ParticipantID'])
    df_cvxh_metrics_ZSB['CH_Area'] = 0
    df_cvxh_metrics_ZSB['CH_Perim'] = 0

    # df with all participant solutions that happend in the post moment
    df_ZSB = df.iloc[ZSB_UKN_lst]

    #! loop to create participants' cvxh for the POST INTERVENTION part and save all in the same image
    for i in range(len(pt_unique)):
        rowsPT = df_ZSB[df_ZSB['ParticipantID'] == pt_unique[i]] # get the lines corresponding to each participant, FROM THE DF_KYY
        rowsPT_idx = rowsPT.index.tolist()

    # getting participant corresponding points from UMAP embedding
        all_x_points = embed[:,0]
        all_y_points = embed[:,1]

        x_points = all_x_points[rowsPT_idx]
        y_points = all_y_points[rowsPT_idx]

        x_y_points = np.array(list(zip(x_points,y_points)))

    # create convexhull for the participant
        hullPT = ConvexHull(x_y_points)
        pts = x_y_points[hullPT.vertices]

        if save_plot == True:
        # set figure size & basic params
            plt.rcParams["figure.figsize"] = (15,15)
            fig, ax = plt.subplots()

        # add points as scatter plot, from x,y of the UMAP embedding ONLY FOR THE ZSB
            scatter = ax.scatter(embed[ZSB_UKN_lst, 0],embed[ZSB_UKN_lst, 1], c=part_c, s=10)

        # get centre & vertices from convex hull
            cent = np.mean(x_y_points, axis=0)
            pts = x_y_points[hullPT.vertices]

            for simplex in hullPT.simplices:
                plt.plot(x_y_points[simplex, 0], x_y_points[simplex, 1], c_ZSB.loc[i,'HEX-AreaCH']) #transparent markers, with solid edge

            k = 1
            color = c_ZSB.loc[i,'HEX-AreaCH']
            poly = Polygon(k*(pts - cent) + cent, closed=True,
                        capstyle='round', facecolor=color, alpha=0.4)
            plt.gca().add_patch(poly)

    # get metrics from convex hull created
        df_cvxh_metrics_ZSB.at[int(df_cvxh_metrics_ZSB[df_cvxh_metrics_ZSB['ParticipantID']== pt_unique[i]].index.values), 'CH_Area'] = hullPT.volume
        df_cvxh_metrics_ZSB.at[int(df_cvxh_metrics_ZSB[df_cvxh_metrics_ZSB['ParticipantID']== pt_unique[i]].index.values), 'CH_Perim'] = hullPT.area
        pt_cvxh.at[int(pt_cvxh[pt_cvxh['ParticipantID'] == pt_unique[i]].index.values), 'ZSB_CH_Vert'] = pts

        #! FULL DESIGN SPACE CONVEX HULL
        # create convexhull for the full UMAP set of points
        hullDS = ConvexHull(embed)

        # get metrics from convex hull created
        DS_area = hullDS.volume
        DS_perim = hullDS.area

        if save_plot == True:
            # get centre & vertices from convex hull
            cent = np.mean(embed, axis=0)
            pts = embed[hullDS.vertices]

            for simplex in hullDS.simplices:
                plt.plot(embed[simplex, 0], embed[simplex, 1], 'k') # edge

            plt.plot(embed[hullDS.vertices, 0], embed[hullDS.vertices, 1], 'o', mec='k', color='none', lw=1, markersize=10) #transparent markers, with solid edge

            k = 1
            color = 'k'
            poly = Polygon(k*(pts - cent) + cent, closed=True,
                        capstyle='round', facecolor=color, alpha=0.005)
            plt.gca().add_patch(poly)

            '''# add labels to the points
            for label, x, y in zip(labels, embedding_umap[:,0], embedding_umap[:,1]):
                plt.annotate(label, (x,y), xycoords = 'data')'''

            if mult_plot is not None:
                # plot annotations
                plt.title(f'DS-Viz ({pt_unique[i]} | 256 Part | 100 gallery), Post, all results, {mult_plot}', fontsize=16, fontweight='bold')
            else:
                plt.title(f'DS-Viz ({pt_unique[i]} | 256 Part | 100 gallery), Post, all results', fontsize=16, fontweight='bold')

        #plt.show()
        if save_plot == True:
            if mult_plot is not None:
                plt.savefig(f'{dir_viz}/DS_{pt_unique[i]}_{mult_plot}_Post.png', dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.savefig(f'{dir_viz}/DS_{pt_unique[i]}_Post.png', dpi=300, bbox_inches='tight')
                plt.close()

    df_DS_vertx['x'] = embed[hullDS.vertices, 0]
    df_DS_vertx['y'] = embed[hullDS.vertices, 1]

    return DS_area, pt_cvxh, df_cvxh_metrics_ZSB, df_DS_vertx

def create_cvxh(dir_data, df, embed, df_colors, mode='all', save_plot=False, mult_plot=None):
    """Create the convex hull for the participants.

    Args:
        dir_data (path): Path to the directory with the data file.
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        embed (matrix): Full embedding with x,y points coordinates.
        df_colors (dataframe): Dataframe with color scheme for the participants.
        mode (str, optional): The focus of the calculation of the convex hull (all, pre, post, full). Defaults to 'all' for plotting pre, post and full session convex hulls.
        save_plot (bool, optional): Parameter that defines if plots will be created and saved. Defaults to False.
        mult_plot (string, optional): Parameter for multi plot naming when validating. Defaults to None.

    Raises:
        ValueError: Mode must be one of {all, pre, post, full}.

    Returns:
        DS_area (float): Area of the DS of all solutions (participants & gallery).
        pt_cvxh (dataframe): Dataframe with convex hull metrics for each participant.
        df_cvxh_metrics (dataframe): Dataframe with convex hull metrics for the full session for each participant.
    """
    #* error if mode is invalid
    if mode not in VALID_MODE:
        raise ValueError("mode: mode must be one of %r." % VALID_MODE)

    #* umap/cvxhull prep
    # KYY + ZSB OVERLAP DF --> contais all points of the cvxhull for each participant
    # will receive the points when running the cvx hulls for KYY/ZSB
    pt_cvxh = df[['ParticipantID']].drop_duplicates().reset_index(drop=True)
    pt_cvxh['KYY_CH_Vert'] = ''
    pt_cvxh['ZSB_CH_Vert'] = ''
    pt_cvxh['FS_CH_Vert'] = ''

    df_DS_vertx = pd.DataFrame(columns=['x', 'y'])

    match mode:
        case 'all':                                     #* call all plots functions for each participant
            print(f"Calculating participants' convex hull for ALL, save = {save_plot}...")
            DS_area, pt_cvxh, df_cvxh_metrics, df_DS_vertx = plot_full(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot, mult_plot)
            DS_area, pt_cvxh, df_cvxh_metrics_KYY, df_DS_vertx = plot_pre(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot, mult_plot)
            DS_area, pt_cvxh, df_cvxh_metrics_ZSB, df_DS_vertx = plot_post(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot, mult_plot)


            print(f"DONE: Calculating participants' convex hull for ALL, save = {save_plot}!")
            return DS_area, pt_cvxh, df_cvxh_metrics, df_cvxh_metrics_KYY, df_cvxh_metrics_ZSB, df_DS_vertx

        case 'full':                                    #* call plot_full only (to plot only the full session DS for each participant)
            DS_area, pt_cvxh, df_cvxh_metrics, df_DS_vertx = plot_full(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot, mult_plot)
            df_cvxh_metrics_KYY = None
            df_cvxh_metrics_ZSB = None

            print(f"DONE: Calculating participants' convex hull for the FULL SESSION, save = {save_plot}!")
            return DS_area, pt_cvxh, df_cvxh_metrics, df_cvxh_metrics_KYY, df_cvxh_metrics_ZSB, df_DS_vertx

        case 'pre':                                     #* call plot_pre only (to plot only the pre intervention part of the session DS for each participant)
            DS_area, pt_cvxh, df_cvxh_metrics_KYY, df_DS_vertx = plot_pre(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot, mult_plot)
            df_cvxh_metrics = None
            df_cvxh_metrics_ZSB = None

            print(f"DONE: Calculating participants' convex hull for the PRE INTERVENTION part of the session, save = {save_plot}!")
            return DS_area, pt_cvxh, df_cvxh_metrics, df_cvxh_metrics_KYY, df_cvxh_metrics_ZSB, df_DS_vertx

        case 'post':                                    #* call plot_post only (to plot only the post intervention part of the session DS for each participant)
            DS_area, pt_cvxh, df_cvxh_metrics_ZSB, df_DS_vertx = plot_post(dir_data, df, pt_cvxh, embed, df_colors, df_DS_vertx, save_plot, mult_plot)
            df_cvxh_metrics = None
            df_cvxh_metrics_KYY = None

            print(f"DONE: Calculating participants' convex hull for the POST INTERVENTION part of the session, save = {save_plot}!")
            return DS_area, pt_cvxh, df_cvxh_metrics, df_cvxh_metrics_KYY, df_cvxh_metrics_ZSB, df_DS_vertx
