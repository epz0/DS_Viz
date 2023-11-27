"""The validation_distmetric runs the comparisons across different weights to see how solutions get spaced out.

Functions:
    get_indices_of_k_smallest: Function to reorder the distance matrix and get the points at the smallest distance from each other.
    get_indices_of_k_largest: Function to reorder the distance matrix and get the points at the largest distance from each other.
    drop_duplicated_points: Function to drop duplicated points from a list of points [(a,b), (c,d), (b,a)] --> [(a,b), (c,d)].
    get_weights: Function to get the weights for the Gower/Jaccard distances run.
    metrics_weights_comparison_specificpos: Runs the comparison accross the different weights comparing the solutions that are at specific distances from each other.
    metrics_weights_comparison_specificsols: Runs the comparison of the specific solutions distances across the different m etrics weights.
    distance_hidim_embed_distort: Runs the distortion between
"""

from pathlib import Path

import ast
import pandas as pd
import numpy as np
import seaborn as sns

from design_space.read_data import read_analysis
from design_space.dist_matrix import *
from design_space.dim_reduction import *
from design_space.design_space import *
from validation.validation_distmetric import *



def prep_plot(df, embed):
    """Returns the limits that will be used in the plotting.

    Args:
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        embed (matrix): Full embedding with x,y points coordinates.

    Returns:
        lim_x: Limits for the x axis.
        lim_y: Limits for the y axis.
    """
    # make df_kde out of the df,embed
    df_kde = df[['ParticipantID', 'GroupID', 'PrePost', 'SolutionID', 'result', 'type']].copy()
    df_kde['x_emb'] = embed[:,0]
    df_kde['y_emb'] = embed[:,1]

    #* plot scatter plot to get limits of x, y
    fig = sns.scatterplot(data=df_kde, x="x_emb", y="y_emb", hue='ParticipantID')
    xmin, xmax = fig.get_xlim()
    ymin, ymax = fig.get_ylim()
    plt.close
    lim_x = [xmin, xmax]
    lim_y = [ymin, ymax]

    return lim_x, lim_y

def multiplot_stp(x_lim, y_lim):
    """Returns the ax to be used for the construction of the multiplot.

    Args:
        x_lim (list): Limits for the x axis.
        y_lim (list): Limits for the y axis.

    Returns:
        axbase: Axis that wil be used to construct the multiplot.
    """
    fig1, axbase = plt.subplots(layout='constrained')
    axbase.set_xlim(x_lim)
    axbase.set_ylim(y_lim)
    axbase.set_xticks([])
    axbase.set_yticks([])
    return axbase


def plot_spec_sols_difweights(dir_data, filenm, df, list_sols, df_colors, embed=None, sheetgow=None, sheetjac=None, save_val=True):
    """PLots a list of solutions across the different metrics weights.

    Args:
        dir_data (path): Path to the directory with the data file.
        filenm (string): Name of the file containing the data for the metrics' weights. Expects Excel file.
        df (dataframe): Dataframe with the analysis for the solutions.
        list_sols (list): List of solutions that will be be compared across the different metrics.
        sheetgow (string): Sheet name with data for the Gower distance weights. Defaults to None.
        sheetjac (string): Sheet name with data for the Jaccard distance weights. Defaults to None.
        save_val (bool, optional): Argument to specify if the comparison should be saved. Defaults to True.
    """

    if embed is not None:
        xl, yl = prep_plot(df, embed)
        axb = multiplot_stp(xl, yl)
        # set figure size & basic params
        fig, ax = plt.subplots(3,4, layout='constrained', figsize=(30, 40), sharey=axb, sharex=axb)



    df_gow_weights, df_jac_weights = get_weights(dir_data, filenm)

    g_weigthtype = []                           #* list of Gower metric weights
    j_weightval = []                            #* list of Jaccard metric weights

    #* transform list of solutions names into list of indices
    list_idx = []
    for sol in list_sols:
        list_idx.append(df[df['FullID']==sol].index.values[0])

        # set figure size & basic params
        #fig, ax = plt.subplots(3,4, layout='constrained', figsize=(30, 40))

    print('Plotting list of solutions in different Gower/Jaccard weights UMAP...')
    #* get weights and the matrices if already exist weights
    for m in range(len(df_gow_weights['gowerW'])):
        wgts_g = ast.literal_eval(df_gow_weights['gowerW'][m])
        wgts_j = df_jac_weights['w_jac'][m]


        g_weigthtype.append(df_gow_weights['Weigth Type'][m])   # list of weight types
        j_weightval.append(wgts_j)                              # list of Jaccard weight values

        #print(g_weigthtype[m])

        #* calculates distance matrix
        n_distmatrix = calc_distmatrix(df, dir_data, filenm, gowerweight=wgts_g, jacweight=wgts_j, mult_gow=m)
        print(f'Dist Matrix with GW: {g_weigthtype[m]}, JW: {j_weightval[m]} calculated.')


        sols_size = len(list_sols)

        #* calculates/retrieves embedding
        embed, emb_graph = create_embedding(dir_data, n_distmatrix, embed_name=(df_gow_weights['WTp'][m]))
        print('embedding done')


        #! PLOT UMAP DIFF VALUES [validation umap parameters]

        ax_cfg = fig.add_subplot(3, 4, (m+1))

        # plot points
        scatter = ax_cfg.scatter(embed[:, 0],embed[:, 1], c='grey') #c=result

        '''# add labels to the points
        for label, x, y in zip(labels, embedding_tsne[:,0], embedding_tsne[:,1]):
            plt.annotate(label, (x,y), xycoords = 'data')
        '''
        #print(list_idx)

        for vtx in list_idx:
                print(vtx)
                ax_cfg.plot(embed[vtx, 0], embed[vtx, 1], 'P', mec='k', color='red', lw=3, markersize=12)

            # add labels
                x = embed[vtx, 0]
                y = embed[vtx, 1]
                ax_cfg.annotate(vtx, (x,y), xycoords = 'data')
                ax_cfg.set_xticklabels([])
                ax_cfg.set_yticklabels([])
    #* title
    fig.suptitle('Different UMAP embeddings weights', fontsize=12, fontweight='bold')
    plt.legend([],[], frameon=False)
    for (i, axi) in zip(range(len(df_gow_weights['WTp'])), ax.flat):
            axi.title.set_text(f'Weight Type {df_gow_weights["WTp"][i]}')
            axi.set_xticklabels([])
            axi.set_yticklabels([])

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.show()