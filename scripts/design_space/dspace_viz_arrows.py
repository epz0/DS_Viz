"""The dspace_viz_arrows module generate the visualisations of the participants' design space exploration, adding the arrows to the plot.

Functions:
    plot_full_ds: Function that plots the full design space (all participants, all parts of the session, and gallery solutions).
    plot_ds_exploration_arrows: Function that plots a single image with the participant exploration of the design space throughout the session (arrows).
    dist_metrics_post: Function that plots a single image *for each solution created by the participant*, highlighting the progression of their space exploration throughout the session.
"""


import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon


def plot_full_ds(embed, sol_c, cvh_plot=True):
    """Returns figure and axis with the scatter plot and convex hull for the full design space.

    Args:
        embed (matrix): Full embedding with x,y points coordinates.
        sol_c (list): List with solution colors for the plot, as specified in the data spreadsheet.
        cvh_plot (bool, optional): If convex hull of the full design space should be traced. Defaults to True.

    Returns:
        fig, ax: Figure and axis elements from matplotlib that will be used downstream.
    """

    #print('Plotting the full design space...')
    #create figure
    plt.rcParams["figure.figsize"] = (15,15)
    fig, ax = plt.subplots()

    # add points as scatter plot, from x,y of the UMAP embedding
    scatter = ax.scatter(embed[:, 0],embed[:, 1], c=sol_c, s=10)

    #! FULL DESIGN SPACE CONVEX HULL
    # create convexhull for the full UMAP set of points
    hullDS = ConvexHull(embed)

    # get metrics from convex hull created
    DS_area = hullDS.volume
    DS_perim = hullDS.area

    if cvh_plot == True:
        # get centre & vertices from convex hull
        cent = np.mean(embed, axis=0)
        pts = embed[hullDS.vertices]

        for simplex in hullDS.simplices:
            ax.plot(embed[simplex, 0], embed[simplex, 1], 'k') # edge
        ax.plot(embed[hullDS.vertices, 0], embed[hullDS.vertices, 1], 'o', mec='k', color='none', lw=1, markersize=10) #transparent markers, with solid edge

        k = 1
        color = 'k'
        poly = Polygon(k*(pts - cent) + cent, closed=True,
                    capstyle='round', facecolor=color, alpha=0.005)
        ax.add_patch(poly)
    #print('Plotting the full design space done!')
    return fig, ax


def plot_ds_exploration_arrows(dir_data, df_unm, embed, df_dist_metrics, sol_c, pt_unique, df_colors, save_plot=True, fs_cvh=True, lbl=False):
    """Plots and saves the DS exploration pattern (arrows) and convex hull of the participants.

    Args:
        dir_data (path): Path to the directory with the data file.
        df_unm (dataframe): Dataframe with all main data, and with the unmasking applied to it.
        embed (matrix): Full embedding with x,y points coordinates.
        df_dist: Dataframe populated with the distance metrics for the full session and pre/post.
        sol_c (list): List with solution colors for the plot, as specified in the data spreadsheet.
        pt_unique (list): List of unique participant IDs.
        df_colors (dataframe): Dataframe with color scheme for the participants.
        save_plot (bool, optional): Parameter that defines if plots will be created and saved. Defaults to True.
        fs_cvh (bool, optional): Parameter that defines if full design spaces should be traced. Defaults to True.
        lbl (bool, optional): Parameter that defines if numerical labels for the solutions should be added. Defaults to False.
    """

    col_ID_PT = 'ParticipantID'                 #! column with the participants' ids
    col_ID_PrePost = 'PrePost'                  #! column with the prepost ids
    col_ID_Sol = 'SolutionID'                   #! column with the solutions ids


    dir_viz = Path(f'{dir_data.parent}'+r'/viz/DS_arrows')

    print('Plotting the DS exploration for each participant...')
    for i in range(len(pt_unique)):
        print(f'Plotting the DS exploration of participant {pt_unique[i]}...')

        plt.close()
        #* call function to plot DS
        f, ax = plot_full_ds(embed, sol_c, cvh_plot=fs_cvh)                   # plot first part of the fig

        #* get the lines corresponding to each participant
        rowsPT = df_unm[df_unm[f'{col_ID_PT}'] == pt_unique[i]]
        rowsPT = rowsPT.index.tolist()
        startPT = rowsPT[0]
        endPT = rowsPT[-1]

        #* get participant colors
        color_area = df_colors.loc[i,'HEX-AreaCH'] #i
        color_edge = df_colors.loc[i,'HEX-Win'] #i

        # getting participant corresponding points from UMAP embedding
        all_x_points = embed[:,0]
        all_y_points = embed[:,1]

        x_points = all_x_points[startPT:endPT+1]
        y_points = all_y_points[startPT:endPT+1]

        x_y_points = np.array(list(zip(x_points,y_points)))                #* all points, x,y array

        #! get ordered list of ids from df_dist (unmasked)
        seq_arrows = df_dist_metrics['ord_idx_FS'][i]
        seq_arrows_pre = df_dist_metrics['ord_idx_PRE'][i]
        seq_arrows_pst = df_dist_metrics['ord_idx_PST'][i]
        first_sol_all = rowsPT.index(seq_arrows_pre[0])     # to look for in the x,y
        first_sol_pst = rowsPT.index(seq_arrows_pst[0])     # to look for in the x,y
        final_sol_all = rowsPT.index(seq_arrows[-1])        # to look for in the x,y

        #! markers, marker color,labels lists
        result = df_unm['result'][startPT:endPT+1]                  # list of results
        prepost = df_unm[f'{col_ID_PrePost}'][startPT:endPT+1]     # list of pre/post types
        labelPT = df_unm[f'{col_ID_Sol}'][startPT:endPT+1]         # getting the labels for the current PT
        labelPT = [int(a) for a in labelPT]                         # float to int

        m_res = ['x' if res=='fail' else '.' for res in result]
        m_pp = ['s' if pp=='Pre' else 'o' for pp in prepost]
        c_res = ['k' if res=='fail' else color_edge for res in result]

        #* matching idx (index of the solution ordered in the unordered list)
        matched_idx = []
        for id in seq_arrows:
            matched_idx.append(rowsPT.index(id))

        if lbl is True:
            # add labels to the points
            for label, x, y in zip(labelPT, x_points, y_points):
                ax.annotate(label, (x,y), xycoords = 'data')

        # create convexhull for the participant
        hullPT = ConvexHull(x_y_points)
        pts = x_y_points[hullPT.vertices]

        # get centre & vertices from convex hull
        cent = np.mean(x_y_points, axis=0)
        pts = x_y_points[hullPT.vertices]

        #! PLOT PARTICIPANT CVX HULL
        for simplex in hullPT.simplices:
            ax.plot(x_y_points[simplex, 0], x_y_points[simplex, 1], color_area) #transparent markers, with solid edge

        k = 1
        color = color_area
        poly = Polygon(k*(pts - cent) + cent, closed=True,
                    capstyle='round', facecolor=color, alpha=0.3)
        ax.add_patch(poly)

        #! PLOT ARROWS
        for j, sol in enumerate(seq_arrows[:-1]):
            arrStartX = all_x_points[sol]
            arrStartY = all_y_points[sol]
            arrEndX = all_x_points[seq_arrows[j+1]]
            arrEndY =all_y_points[seq_arrows[j+1]]

            ax.arrow(arrStartX,arrStartY,arrEndX-arrStartX, arrEndY-arrStartY,
                    shape='full', color=color_edge,  length_includes_head=True,
                    zorder=0, head_length=.1, head_width=.055, lw=1.5 )

        first_sol_all = rowsPT.index(seq_arrows_pre[0])     # to look for in the x,y
        first_sol_pst = rowsPT.index(seq_arrows_pst[0])     # to look for in the x,y
        final_sol_all = rowsPT.index(seq_arrows[-1])

        # add markers style for participant
        # m_result
        for x, y, m, c in zip(x_y_points[:, 0], x_y_points[:, 1], m_res, c_res):
            ax.plot(x,y, marker=m, color=c, linewidth=6, markersize=10)

        # m_prepost
        for x, y, m in zip(x_y_points[:, 0], x_y_points[:, 1], m_pp):
            #ax.scatter(x, y , marker=m, edgecolors=color_edge, color='none', linewidths=5, s=15)
            ax.plot(x, y, marker=m, mec=color_edge, color='none', linewidth=3, markersize=12) #transparent markers, with solid edge

        # just bigger markers
        #for x, y in zip(x_y_points[:, 0], x_y_points[:, 1]):
        #    ax.plot(x, y, marker='.', mec=color_edge, color=color_edge, markersize=13)

        #* start style (*)
        ax.plot(x_y_points[first_sol_all, 0], x_y_points[first_sol_all, 1], '*', mec=color_edge, color='k', lw=1, markersize=18)

        #* first pst style (d)
        ax.plot(x_y_points[first_sol_pst, 0], x_y_points[first_sol_pst, 1], 'P', mec=color_edge, color='k', lw=1, markersize=15)

        #* last style (s)
        ax.plot(x_y_points[final_sol_all, 0], x_y_points[final_sol_all, 1], 'X', mec=color_edge, color='k', lw=1, markersize=15)

        # plot annotations
        plt.axis('on')
        plt.title(f'{pt_unique[i]} DS Exploration', fontsize=16, fontweight='bold')
        #plt.show()

        if save_plot == True:
            plt.savefig(f'{dir_viz}/DS_{pt_unique[i]}_Exploration_FS.png', dpi=300, bbox_inches='tight')
            print(f'saved {pt_unique[i]}')

        print(f'Plotting the DS exploration of participant {pt_unique[i]} (save={save_plot}) done!')

    print(f'Plotting the DS exploration done!')


def plot_ds_exploration_incremental_arrows(dir_data, df_unm, embed, df_dist_metrics, sol_c, pt_unique, df_colors, save_plot=True, fs_cvh=True, lbl=False):
    """Plots and saves the incremental DS exploration pattern (arrows) and convex hull of the participants.

    Args:
        dir_data (path): Path to the directory with the data file.
        df_unm (dataframe): Dataframe with all main data, and with the unmasking applied to it.
        embed (matrix): Full embedding with x,y points coordinates.
        df_dist: Dataframe populated with the distance metrics for the full session and pre/post.
        sol_c (list): List with solution colors for the plot, as specified in the data spreadsheet.
        pt_unique (list): List of unique participant IDs.
        df_colors (dataframe): Dataframe with color scheme for the participants.
        save_plot (bool, optional): Parameter that defines if plots will be created and saved. Defaults to True.
        fs_cvh (bool, optional): Parameter that defines if full design spaces should be traced. Defaults to True.
        lbl (bool, optional): Parameter that defines if numerical labels for the solutions should be added. Defaults to False.
    """

    col_ID_PT = 'ParticipantID'                 #! column with the participants' ids
    col_ID_PrePost = 'PrePost'                  #! column with the prepost ids
    col_ID_Sol = 'SolutionID'                   #! column with the solutions ids

    dir_viz = Path(f'{dir_data.parent}'+r'/viz/DS_arrows')

    print('Plotting the incremental DS exploration for each participant...')
    for i in range(len(pt_unique)):
        print(f'Plotting the incremental DS exploration of participant {pt_unique[i]}...')
        # getting participant corresponding points from UMAP embedding
        all_x_points = embed[:,0]
        all_y_points = embed[:,1]

        #* get the lines corresponding to each participant
        rowsPT = df_unm[df_unm[f'{col_ID_PT}'] == pt_unique[i]]
        rowsPT = rowsPT.index.tolist()
        startPT = rowsPT[0]

        #* get participant colors
        color_area = df_colors.loc[i,'HEX-AreaCH'] #i
        color_edge = df_colors.loc[i,'HEX-Win'] #i

        #! get ordered list of ids from df_dist (unmasked)
        seq_arrows = df_dist_metrics['ord_idx_FS'][i]
        seq_arrows_pre = df_dist_metrics['ord_idx_PRE'][i]
        seq_arrows_pst = df_dist_metrics['ord_idx_PST'][i]

        #* matching idx (index of the solution ordered in the unordered list)
        matched_idx = []
        for id in seq_arrows:
            matched_idx.append(rowsPT.index(id))

        #print('seq', seq_arrows)
        #print('matched',matched_idx)
        #print('rowspt', rowsPT)

        #print(startPT, rowsPT, f'len {len(rowsPT)}')

        #*REORDER X_Y POINTS
        #print('all', all_x_points, all_y_points)
        endPT_all = startPT+len(rowsPT)+1
        all_x_unord = all_x_points[startPT:endPT_all]
        #print('x', all_x_unord)

        all_y_unord = all_y_points[startPT:endPT_all]
        #print('y', all_y_unord)

        all_x_y_unord = np.array(list(zip(all_x_unord,all_y_unord)))
        #print('all un', all_x_y_unord)

        x_y_points_ord = np.array([all_x_y_unord[z] for z in matched_idx])
        #print('all ord', x_y_points_ord)

        first_sol_all = rowsPT.index(seq_arrows_pre[0])     # to look for in the x,y
        first_sol_pst = len(seq_arrows_pre)+1               # gets the position of the first pst in the ordered x,y list
        final_sol_all = rowsPT.index(seq_arrows[-1])        # to look for in the x,y

        totalsol = len(rowsPT)
        #! markers, marker color,labels lists
        result = list(df_unm['result'][startPT:totalsol+startPT])               # list of results
        prepost = list(df_unm[f'{col_ID_PrePost}'][startPT:totalsol+startPT])     # list of pre/post types
        labelPT = list(df_unm[f'{col_ID_Sol}'][startPT:totalsol+startPT])         # getting the labels for the current PT
        labelPT = [int(a) for a in labelPT]                          # float to int
        #print(f'resul pre ord {result}', type(result))
        #print('pre ord', labelPT)
        #print('result', result)

        #! reorder markers lists
        result = [result[z] for z in matched_idx]
        prepost = [prepost[z] for z in matched_idx]
        labelPT = [labelPT[z] for z in matched_idx]
        #print(f'resul post ord {result}', type(result))
        #print('post ord', labelPT)
        endPT_arr = 0
        for n in range(len(rowsPT)):
            plt.close()
            #* call function to plot DS
            f, ax = plot_full_ds(embed, sol_c, cvh_plot=fs_cvh)                   # plot first part of the fig

            #print(rowsPT)

            #DEFINING END POINT
            endPT_arr = rowsPT[n]+1
            #print(' end pt', endPT_arr)
            #print(' startPT', startPT)

            # subsetting list of points
            x_points = x_y_points_ord[0:n+1,0]
            y_points = x_y_points_ord[0:n+1,1]

            x_y_points = np.array(list(zip(x_points,y_points)))                #* all points, x,y array, ORDERED
            #print('xy order', x_y_points)

            m_res = ['x' if res=='fail' else '.' for res in result[0:n+1]]
            m_pp = ['s' if pp=='Pre' else 'o' for pp in prepost[0:n+1]]
            c_res = ['k' if res=='fail' else color_edge for res in result[0:n+1]]
            labelSS = labelPT[0:n+1]
            #print(labelSS, x_points)
            #print(m_res, m_pp, c_res)

            if lbl is True:
                # add labels to the points
                for label, x, y in zip(labelSS, x_points, y_points):
                    ax.annotate(label, (x,y), xycoords = 'data')
            # print(x_y_points)
            match n:
                case 0:
                    #* start style (X)
                    ax.plot(x_y_points[0, 0], x_y_points[0, 1], '*', mec=color_edge, color='k', lw=2, markersize=18)

                    # plot annotations
                    #plt.axis('on')
                    #plt.title(f'DS-Viz ({pt_unique[i]} - Sol{n+1}), FS, all results', fontsize=16, fontweight='bold')
                    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

                    if save_plot == True:
                        plt.savefig(f'{dir_viz}/{pt_unique[i]}_S_0{n+1}.png', dpi=300, bbox_inches='tight')
                        print(f'saved {pt_unique[i]} Sol-{n+1}')
                case 1:
                    #* start style (X)
                    ax.plot(x_y_points[0, 0], x_y_points[0, 1], '*', mec=color_edge, color='k', lw=2, markersize=18)

                    #! PLOT ARROWS
                    for j in range(n):
                        arrStartX = x_y_points[j,0]
                        arrStartY = x_y_points[j,1]
                        arrEndX = x_y_points[j+1,0]
                        arrEndY = x_y_points[j+1,1]

                        ax.arrow(arrStartX,arrStartY,arrEndX-arrStartX, arrEndY-arrStartY,
                        shape='full', color=color_edge,  length_includes_head=True,
                        zorder=0, head_length=.1, head_width=.055, lw=1.5 )

                    #! STYLE MARKERS
                    # add markers style for participant
                    # m_result
                    for x, y, m, c in zip(x_y_points[:, 0], x_y_points[:, 1], m_res, c_res):
                        ax.plot(x,y, marker=m, color=c, linewidth=6, markersize=10)

                    # m_prepost
                    for x, y, m in zip(x_y_points[:, 0], x_y_points[:, 1], m_pp):
                        #ax.scatter(x, y , marker=m, edgecolors=color_edge, color='none', linewidths=5, s=15)
                        ax.plot(x, y, marker=m, mec=color_edge, color='none', linewidth=3, markersize=12) #transparent markers, with solid edge

                    # just bigger markers
                    #for x, y in zip(x_y_points[:, 0], x_y_points[:, 1]):
                    #    ax.plot(x, y, marker='.', mec=color_edge, color=color_edge, markersize=13)

                    #* PLOT PARAMS
                    #plt.title(f'DS-Viz ({pt_unique[i]} - Sol{n+1}), FS, all results', fontsize=16, fontweight='bold')
                    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

                    if save_plot == True:
                        plt.savefig(f'{dir_viz}/{pt_unique[i]}_S_0{n+1}.png', dpi=300, bbox_inches='tight')
                        print(f'saved {pt_unique[i]} Sol-{n+1}')
                case 2:
                    #* start style (X)
                    ax.plot(x_y_points[0, 0], x_y_points[0, 1], '*', mec=color_edge, color='k', lw=2, markersize=18)

                    #! PLOT ARROWS
                    for j in range(n):
                        arrStartX = x_y_points[j,0]
                        arrStartY = x_y_points[j,1]
                        arrEndX = x_y_points[j+1,0]
                        arrEndY = x_y_points[j+1,1]

                        ax.arrow(arrStartX,arrStartY,arrEndX-arrStartX, arrEndY-arrStartY,
                        shape='full', color=color_edge,  length_includes_head=True,
                        zorder=0, head_length=.1, head_width=.055, lw=1.5 )

                    #! ==== PLOT CVX HULL ======
                    # create convexhull for the participant
                    hullPT = ConvexHull(x_y_points)
                    pts = x_y_points[hullPT.vertices]

                    # get centre & vertices from convex hull
                    cent = np.mean(x_y_points, axis=0)
                    pts = x_y_points[hullPT.vertices]

                    for simplex in hullPT.simplices:
                        ax.plot(x_y_points[simplex, 0], x_y_points[simplex, 1], color_area) #transparent markers, with solid edge

                    k = 1
                    color = color_area
                    poly = Polygon(k*(pts - cent) + cent, closed=True,
                                capstyle='round', facecolor=color, alpha=0.3)
                    ax.add_patch(poly)
                    #! ==== PLOT CVX HULL ======

                    #! STYLE MARKERS
                    # add markers style for participant
                    # m_result
                    for x, y, m, c in zip(x_y_points[:, 0], x_y_points[:, 1], m_res, c_res):
                        ax.plot(x,y, marker=m, color=c, linewidth=6, markersize=10)

                    # m_prepost
                    for x, y, m in zip(x_y_points[:, 0], x_y_points[:, 1], m_pp):
                        #ax.scatter(x, y , marker=m, edgecolors=color_edge, color='none', linewidths=5, s=15)
                        ax.plot(x, y, marker=m, mec=color_edge, color='none', linewidth=3, markersize=12) #transparent markers, with solid edge

                    # just bigger markers
                    #for x, y in zip(x_y_points[:, 0], x_y_points[:, 1]):
                    #    ax.plot(x, y, marker='.', mec=color_edge, color=color_edge, markersize=13)

                    #*PLOT PARAMS
                    #plt.title(f'DS-Viz ({pt_unique[i]} - Sol{n+1}), FS, all results', fontsize=16, fontweight='bold')
                    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

                    if save_plot == True:
                        plt.savefig(f'{dir_viz}/{pt_unique[i]}_S_0{n+1}.png', dpi=300, bbox_inches='tight')
                        print(f'saved {pt_unique[i]} Sol-{n+1}')
                case _:
                    #* start style (X)
                    ax.plot(x_y_points[0, 0], x_y_points[0, 1], '*', mec=color_edge, color='k', lw=2, markersize=18)

                    #! PLOT ARROWS
                    for j in range(n):
                        arrStartX = x_y_points[j,0]
                        arrStartY = x_y_points[j,1]
                        arrEndX = x_y_points[j+1,0]
                        arrEndY = x_y_points[j+1,1]

                        ax.arrow(arrStartX,arrStartY,arrEndX-arrStartX, arrEndY-arrStartY,
                        shape='full', color=color_edge,  length_includes_head=True,
                        zorder=0, head_length=.1, head_width=.055, lw=1.5 )

                    #! ==== PLOT CVX HULL ======
                    # create convexhull for the participant
                    hullPT = ConvexHull(x_y_points)
                    pts = x_y_points[hullPT.vertices]

                    # get centre & vertices from convex hull
                    cent = np.mean(x_y_points, axis=0)
                    pts = x_y_points[hullPT.vertices]

                    for simplex in hullPT.simplices:
                        ax.plot(x_y_points[simplex, 0], x_y_points[simplex, 1], color_area) #transparent markers, with solid edge

                    k = 1
                    color = color_area
                    poly = Polygon(k*(pts - cent) + cent, closed=True,
                                capstyle='round', facecolor=color, alpha=0.3)
                    ax.add_patch(poly)

                    #! STYLE MARKERS
                    # add markers style for participant
                    # m_result
                    for x, y, m, c in zip(x_y_points[:, 0], x_y_points[:, 1], m_res, c_res):
                        ax.plot(x,y, marker=m, color=c, linewidth=6, markersize=10)

                    # m_prepost
                    for x, y, m in zip(x_y_points[:, 0], x_y_points[:, 1], m_pp):
                        #ax.scatter(x, y , marker=m, edgecolors=color_edge, color='none', linewidths=5, s=15)
                        ax.plot(x, y, marker=m, mec=color_edge, color='none', linewidth=3, markersize=12) #transparent markers, with solid edge

                    # just bigger markers
                    #for x, y in zip(x_y_points[:, 0], x_y_points[:, 1]):
                    #    ax.plot(x, y, marker='.', mec=color_edge, color=color_edge, markersize=13)

                    #! show markers for pst/last if n>pre size
                    if n >= first_sol_pst:
                        #* first pst style (P)
                        ax.plot(x_y_points[first_sol_pst, 0], x_y_points[first_sol_pst, 1], 'P', mec=color_edge, color='k', lw=2, markersize=15)

                    #print(f'totalsol - {totalsol}')
                    if n+1 == totalsol:
                        #print(n+1)
                        #* last style (s)
                        ax.plot(x_y_points[-1, 0], x_y_points[-1, 1], 'X', mec=color_edge, color='k', lw=2, markersize=15)

                    #! ==== PLOT CVX HULL ======

                    #*PLOT PARAMS
                    #plt.title(f'DS-Viz ({pt_unique[i]} - Sol{n+1}), FS, all results', fontsize=16, fontweight='bold')

                    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

                    if save_plot == True:
                            plt.savefig(f'{dir_viz}/{pt_unique[i]}_S_0{n+1}.png', dpi=300, bbox_inches='tight')
                            print(f'saved {pt_unique[i]} Sol-{n+1}')
            endPT=0
        print(f'Plotting the incremental DS exploration of participant {pt_unique[i]} (save={save_plot}) done!')
    print('Plotting the incremental DS exploration for each participant done!')