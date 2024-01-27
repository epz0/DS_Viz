"""The validation_run module supports the validation of different metrics/choices.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from design_space.read_data import *
from design_space.dist_matrix import *
from design_space.dim_reduction import *

from validation.validation_distmetric import *
from validation.validation_areametric import *
from validation.validation_viz import *
from interactive_run import *

#* --- initial definitions ------
my_dir = Path(r'C:/Py/DS_Viz/data')                                              # path to the data file
filenm = 'dataset.xlsx'                                                          # name of the data file
sheetnm = 'data'                                                                 # sheet where the data (analysis) is
#* ------------------------------

#* --- create folder structure ---
Path(f'{my_dir.parent}/validation').mkdir(parents=True,exist_ok=True)            # folder export/stats
#* ------------------------------

#! ---------- DISTANCE WEIGHTS VALIDATION ----------
#* 1. reads Excel file with solutions' analysis
df_base, df_colors, labels = read_analysis(my_dir, filenm, sheetnm)

#* calculates distance matrix
n_distmatrix = calc_distmatrix(df_base, my_dir, filenm)

#* generate the embedding of the distance matrix
embedding, graph = create_embedding(my_dir, n_distmatrix)

#* 2. run comparison specific distance positions [0, 0.25, 0.5, 0.75, 1]
metrics_weights_comparison_specificpos(my_dir, filenm, df_base)

#* 3. run comparison for specific list of solutions
ls_sols = [ 'P_00k-Pre-08',
            'P_00x-Pre-03',
            'P_00x-Pst-16',
            'P_00y-Pst-17',
            'w87kx',
            'qeG7o',
            '4Ld2q',
            'Ag5q6',
            '9ZDod'
        ]

metrics_weights_comparison_specificsols(my_dir, filenm, df_base,ls_sols)

#* 4. distort hidim/embed validation [takes a long time (~1h30 for 2800 combinations)]
distance_hidim_embed_distort(my_dir, filenm, df_base, mult_runs=True)

#! ---------- AREA SENSITIVITY VALIDATION ----------
area_sensitivity_weights(my_dir, filenm, df_base, df_colors)

#! ---------- SOLUTIONS IN PLOT VIZ VALIDATION ----------
plot_spec_sols_difweights(my_dir, filenm, df_base, ls_sols, df_colors, embed=embedding)

#! ---- EVALUATION PARAMETERS EFFECT IN EMBEDDING ----
# specify baseline parameters
wgt = 'W3'
nn = 60
md = 0.15
ds_l = 2

# create baseline parameter plot
embedding, graph = create_embedding(my_dir, n_distmatrix, Wg=wgt, NN=nn, MD=md, densm=ds_l)
plot_embedding(my_dir, embedding, df_base, embed_name=f'{wgt}-NN_{nn}-MD_{md}-dens_{ds_l}')
create_cvxh(my_dir, df_base, embedding, df_colors, mode='all', save_plot=True)

# get list of cvx hull vertices
x_val = embedding[:, 0]
y_val = embedding[:, 1]

x_vtx, y_vtx, cvxh_area = cv_hull_vertices(x_val, y_val)

vtx = list(zip(x_vtx,y_vtx))
emb_lst = list(zip(x_val,y_val))

# get solution ids for the vertice points
ls_ids_vtx =[]
pts_vtx =[]

for vt in vtx:
    ls_ids_vtx.append(emb_lst.index(vt))
    pts_vtx.append(df_base.loc[emb_lst.index(vt),'FullID'])

# reduce list of interest
rem_ids = [2,4,6,8]
ls_ids_vtx = np.delete(np.array(ls_ids_vtx), rem_ids).tolist()
pts_vtx = np.delete(np.array(pts_vtx), rem_ids).tolist()


#* generate multiplot with varying NN and MD values (plots might be small due to varying axes)
#! multiplot setup
xl, yl = prep_plot(df_base, embedding)
axb = multiplot_stp(xl, yl)
fig, ax = plt.subplots(3,5, layout='tight', figsize=(20, 12), sharey=True, sharex=True)

#* definition of parameters range

# NN list
nn_ls = [35, 40, 45, 50, 55]

# MD list
md_ls = [0.10, 0.20, 0.25]

# retrieving/calculating embeddings and plotting, highlighting points of interest.
for i in range(3):
    for j in range(5):
        axi = ax[i,j]
        emb_param, graph = create_embedding(my_dir, n_distmatrix, Wg=wgt, NN=nn_ls[j], MD=md_ls[i], densm=ds_l)
        axi.scatter(emb_param[:, 0],emb_param[:, 1], c='grey')

        for vtx in ls_ids_vtx:
                print(vtx)
                axi.plot(emb_param[vtx, 0], emb_param[vtx, 1], 'P', mec='k', color='red', lw=3, markersize=12)

                # add labels
                x = emb_param[vtx, 0]
                y = emb_param[vtx, 1]
                axi.annotate(vtx, (x,y), xycoords = 'data')

        axi.set_title(f'NN={nn_ls[j]}, MD={md_ls[i]}')

# Adjust layout
plt.tight_layout()
# Show plot
plt.show()
plt.close()

#* generate multiple individual plots varying NN and MD values (plots are larger)
#! multiple individual plots setup
'''
#NN list
nn_ls = [35, 40, 45, 50, 55]

#MD list
md_ls = [0.10, 0.20, 0.25]

# retrieving/calculating embeddings and plotting, highlighting points of interest.
for nn in nn_ls:
    for md in md_ls:
        fig, ax = plt.subplots(layout='tight', figsize=(5, 5))

        emb_param, graph = create_embedding(my_dir, n_distmatrix, Wg=wgt, NN=nn, MD=md, densm=ds_l)

        plt.scatter(emb_param[:, 0],emb_param[:, 1], c='grey')

        # create convexhull for the full UMAP set of points
        hullDS = ConvexHull(emb_param)

        # get metrics from convex hull created
        DS_area = hullDS.volume
        DS_perim = hullDS.area

        # get centre & vertices from convex hull
        cent = np.mean(emb_param, axis=0)
        pts = emb_param[hullDS.vertices]

        for simplex in hullDS.simplices:
            plt.plot(emb_param[simplex, 0], emb_param[simplex, 1], 'k') # edge

        plt.plot(emb_param[hullDS.vertices, 0], emb_param[hullDS.vertices, 1], 'o', mec='k', color='none', lw=1, markersize=10) #transparent markers, with solid edge

        for vtx in ls_ids_vtx:
                print(vtx)
                plt.plot(emb_param[vtx, 0], emb_param[vtx, 1], 'P', mec='k', color='red', lw=3, markersize=20)

            # add labels
                x = emb_param[vtx, 0]
                y = emb_param[vtx, 1]
                plt.annotate(vtx, (x+0.12,y), xycoords = 'data')

        ax.set_title(f'NN={nn}, MD={md}')

        dir_viz = Path(f'{my_dir.parent}'+r'/viz')
        plt.savefig(f'{dir_viz}/NN={nn}_MD={md}.png', dpi=300, format='png')
        plt.show()
        plt.close()
'''