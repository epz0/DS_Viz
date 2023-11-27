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

#* --- initial definitions ------
my_dir = Path(r'C:/DS_Viz/data')                                                   # path to the data file
filenm = 'dataset.xlsx'                                             # name of the data file
sheetnm = 'data'                                                   # sheet where the data (analysis) is
#* ------------------------------

#* --- create folder structure ---
Path(f'{my_dir.parent}/validation').mkdir(parents=True,exist_ok=True)             # folder export/stats
#* ------------------------------

#! ---------- DISTANCE WEIGHTS VALIDATION ----------
#* procedure:
#       1. get df with solutions' analysis                                                           (read_analysis)
#       2. run comparison of metrics'weights retrieving pairs of solutions' at specific distances    (metrics_weights_comparison_specificpos)
#       3. run comparison of metrics'weights for a specific list of solutions                        (metrics_weights_comparison_specificsols)
#       4. run distortion hidim/embedding for each weight, set of UMAP parameters                    ()

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
            #'w87kx',
            #'qeG7o',
            #'4Ld2q',
            #'Ag5q6',
            #'9ZDod'
        ]

#metrics_weights_comparison_specificsols(my_dir, filenm, df_base,ls_sols)
#
#* 4. distort hidim/embed validation [takes long (~1h30 for 2800 combinations)]
#distance_hidim_embed_distort(my_dir, filenm, df_base, mult_runs=True)

#! ---------- AREA SENSITIVITY VALIDATION ----------
#area_sensitivity_weights(my_dir, filenm, df_base, df_colors)

#! ---------- SOLUTIONS IN PLOT VIZ VALIDATION ----------
plot_spec_sols_difweights(my_dir, filenm, df_base, ls_sols, df_colors, embed=embedding)
