from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from design_space.read_data import read_analysis
from design_space.dist_matrix import *
from design_space.dim_reduction import *
from design_space.dspace_viz_density import *

def novelty_from_neig(dir_data, df, embed, delta, save_df=False, save_plot=True, fn=None):
    """Returns list with novelty scores for each solution, givent a delta (radius).

    Args:
        dir_data (path): Path to the directory with the data file.
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        embed (matrix): Full embedding with x,y points coordinates.
        delta (float): Value for the radius that will be considered for the number of neighbors evaluation.
        save_df (bool, optional): Argument to specify if the dataframe with the novelty metric column should be saved. Defaults to False.
        save_plot (bool, optional): Argument to specify if the density plot with the highlighted points should be created. Defaults to False.

    Returns:
        novel_nn: List with novelty values for each solution.
    """
    dir_ex = Path(f'{dir_data.parent}'+r'/experimental/')
    dir_exviz = Path(f'{dir_data.parent}'+r'/experimental/viz')

    if fn is None:
        fn = f'dt {delta:.2f}'

    df_dens, x_lim, y_lim = prep_density(df, embed)

    print(f'Calculating novelty scores for delta = {delta:.2f}...')
    # create dist matrix from embedding points
    dmatrix_embed = create_dmatrix_from_embed(dir_data, embed, norm=False)

    nn_l = []
    # create number of neighbors list
    for i in range(len(dmatrix_embed)):
        nn_l.append(np.count_nonzero(dmatrix_embed[i] < delta))

    # list with novelty calculation
    novel_nn = [1/item if item > 0 else 2 for item in nn_l]

    if len(df) == len(novel_nn):
        df_dens['novel'] = novel_nn
    else:
        print('Error - lenghts do not match')

    if save_df == True:
        with pd.ExcelWriter(f'{dir_ex}/DS_NoveltyNeighbors_Metric.xlsx') as writer:
                df_dens.to_excel(writer, sheet_name="NoveltyfromNeighbors", index=False)

    print(f'Calculating novelty scores for delta = {delta:.2f} - done!')

    if save_plot == True:
        plt.close()
        topperc = np.percentile(novel_nn, 90)
        botperc = np.percentile(novel_nn, 10)

        df_topten = df_dens.loc[df_dens['novel'] >= topperc].copy()
        df_botten = df_dens.loc[df_dens['novel'] <= botperc].copy()

        sns.set_theme(style="white")
        #sns.set(style="white")#, context="talk")
        #plt.style.use("dark_background")

        fig, axes = plt.subplots(layout='constrained', figsize=(10, 10))
        plot = sns.kdeplot(data=df_dens, x="x_emb", y="y_emb", fill=True, levels=10, cmap='mako')
        sns.scatterplot(data=df_dens, x="x_emb", y="y_emb", color='.2', marker='+', s=50)
        sns.scatterplot(data=df_topten, x="x_emb", y="y_emb", color='red', marker='P', s=75)
        sns.scatterplot(data=df_botten, x="x_emb", y="y_emb", color='k', marker='X', s=75)
        plot.set_xlim(x_lim)
        plot.set_ylim(y_lim)

        plt.legend([],[], frameon=False)
        plt.title(f'Density plot Novelty [NN, r={delta:.2f}] >= 90% (red, +), =< 10% (black, X), full DS', fontsize=14, fontweight='bold')
        axes.set(xlabel=None, ylabel=None)
        plt.savefig(f'{dir_exviz}/EX_Novelty_NN_{fn}.png', dpi=300, bbox_inches='tight')

        print('KDE plot - done!')


    return novel_nn

def novelty_nn_study(dir_data, embed, delta_r=[0.05,1.05], type_hist='probability', x_fixed=True, bin=10):
    """Plots number of neighbors and novelty score given different values of delta (radius).

    Args:
        dir_data (_type_): _description_
        embed (matrix): Full embedding with x,y points coordinates.
        delta_r (list, optional): List with initial and final values of delta range. Defaults to [0.05,1.05].
        type_hist (str, optional): Type of histogram from seaborn desired. Defaults to 'probability'.
        x_fixed (bool, optional): If xticks should be the same across all plots or customised given the different values. Defaults to True.
        bin (integer, optional): Number of bins for the histogram. Defaults to 10
    """
    sns.set_theme(style="white")

    dir_ex = Path(f'{dir_data.parent}'+r'/experimental/')
    dir_exviz = Path(f'{dir_data.parent}'+r'/experimental/viz')
    delta_ran = np.arange(delta_r[0],delta_r[1],0.05)

    # create dist matrix from embedding points
    dmatrix_embed = create_dmatrix_from_embed(dir_data, embed, norm=False)

    print(f'Calculating NN and novelty given delta range {delta_ran[0]}-{delta_ran[-1]}...')
    # initialise lists
    nn_l = []
    nn_lst = []
    bn_lst = []
    novel_lst = []
    bn_nov =[]

    for dt in range(len(delta_ran)):
        nn_l = []

        # create number of neighbors list
        for i in range(len(dmatrix_embed)):
            nn_l.append(np.count_nonzero(dmatrix_embed[i] < delta_ran[dt]))

        # list with all nn_l for each dt
        nn_lst.append(nn_l)

        bn = len(list(dict.fromkeys(nn_l)))                               # number of unique number of neighbors
        # list with number of bins corresponding to each list of NN
        bn_lst.append(bn)

        # list with novelty calculation
        novel_nn = [1/item if item > 0 else 2 for item in nn_l]
        novel_lst.append(novel_nn)

        bn = len(list(dict.fromkeys(novel_nn)))
        bn_nov.append(bn)

    print(f'Calculating NN and novelty given delta range {delta_ran[0]}-{delta_ran[-1]} - done!')

    #* plotting
    sns.set_theme(style="white")

    print(f'Plotting NN and novelty given delta range {delta_ran[0]}-{delta_ran[-1]}...')
    # defining plot region
    fig, axes = plt.subplots(4, 5, layout='constrained', figsize=(20, 25))

    #! plot histogram probability vs NN
    for (n, axi) in zip(range(len(delta_ran)), axes.flat):
        splot = sns.histplot(data=nn_lst[n], ax=axi, stat=f'{type_hist}', kde=True, bins=bin)
        axi.title.set_text(f'delta: {delta_ran[n]:.2f}')
        axi.set(xlabel='Number of Neighbors', ylabel=f'{type_hist}')

        if x_fixed == True:
            splot.set_xlim(0,80)
        else:
            splot.set_xlim(0,max(nn_lst[n]))


    fig.suptitle('Number of Neighbors distribution for each Delta', fontsize=14, fontweight='bold')
    plt.savefig(f'{dir_exviz}/EX_NN-delta-{type_hist}.png', dpi=300, bbox_inches='tight')

    plt.close()
    figg, axess = plt.subplots(4, 5, layout='constrained', figsize=(20, 25))

    for (n, axi) in zip(range(len(delta_ran)), axess.flat):

        splot = sns.histplot(data=novel_lst[n], ax=axi, stat=f'{type_hist}', kde=True, bins=bin, color='palevioletred')
        axi.title.set_text(f'delta: {delta_ran[n]:.2f}')
        axi.set(xlabel='Novelty Score', ylabel=f'{type_hist}',)

        if x_fixed == True:
            splot.set_xlim(0,1)
        else:
            splot.set_xlim(0,max(novel_lst[n]))

    figg.suptitle('Novelty Score distribution for each Delta', fontsize=14, fontweight='bold')
    plt.savefig(f'{dir_ex}/EX_Novelty-NN-delta-{type_hist}.png', dpi=300, bbox_inches='tight')
    print(f'Plotting NN and novelty given delta range {delta_ran[0]}-{delta_ran[-1]} - done!')