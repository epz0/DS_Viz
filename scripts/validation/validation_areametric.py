"""The validation_area module provides functionality to assess the sensitivity of convex hull areas given different distance metrics weights.

Functions:
    area_sensitivity_weights: Runs the validation for assessing areas' sensitivity to different weights.
"""

from pathlib import Path
import plotly.express as px
from validation.validation_distmetric import *
from design_space.read_data import *
from design_space.dist_matrix import *
from design_space.dim_reduction import *
from design_space.design_space import *
from design_space.dspace_metrics import *
from design_space.dspace_dist_metrics import *



def area_sensitivity_weights(dir_data, filenm, df, df_colors, save_plt=True):
    """Runs the validation for assessing areas' sensitivity to different weights.

    Args:
        dir_data (path): Path to the directory with the data file.
        fname (string): Name of the file containing the data for the metrics' weights. Expects Excel file.
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        df_colors (dataframe): Dataframe with participants' color scheme.
        save_plt (bool, optional): Argument to define if plots should be saved. Defaults to False.
    """
    df_gow_weights, df_jac_weights = get_weights(dir_data, filenm)

    dfsAreas_lst = []
    DSAreas_lst =[]
    g_weigthtype = []                           #* list of Gower metric weights
    j_weightval = []                            #* list of Jaccard metric weights


    print('Running area sensitivity analysis for the different Gower/Jaccard weights...')
    #* get metric weights
    for m in range(len(df_gow_weights['gowerW'])):
        wgts_g = ast.literal_eval(df_gow_weights['gowerW'][m])
        wgts_j = df_jac_weights['w_jac'][m]

        g_weigthtype.append(df_gow_weights['Weigth Type'][m])   # list of weight types
        j_weightval.append(wgts_j)                              # list of Jaccard weight values

        #! calculates distance matrix
        n_distmatrix = calc_distmatrix(df, dir_data, filenm, gowerweight=wgts_g, jacweight=wgts_j, mult_gow=m)
        print(f'Dist Matrix with GW: {g_weigthtype[m]}, JW: {j_weightval[m]} calculated.')

        #! create the embedding
        embed, emb_graph = create_embedding(dir_data, n_distmatrix, embed_name=(df_gow_weights['WTp'][m]))
        print('embedding done')

        #! Create convex hulls, get the metrics
        DS_area, pt_cvxh, df_cvxh_metrics, df_cvxh_metrics_KYY, df_cvxh_metrics_ZSB, df_DS_vertx = create_cvxh(dir_data, df, embed, df_colors, mode='all', save_plot=save_plt, mult_plot=(df_gow_weights['WTp'][m]))
        DSAreas_lst.append(DS_area) #! area of full DS
        print('cvx hulls done')

        #! get the summary of the areas for the Pre v Post setting
        df_DScover = area_summary(df, DS_area, df_cvxh_metrics, mode='all',df_ch_pre_metrics=df_cvxh_metrics_KYY, df_ch_post_metrics=df_cvxh_metrics_ZSB)
        dfsAreas_lst.append(df_DScover)
        print('summary areas done')

    #! get core metrics for each participant in a single df
    df_all = pd.DataFrame()
    for i in range(len(dfsAreas_lst)):
        tempdf = dfsAreas_lst[i]
        tempdf = tempdf.rename({ 'Area-KYY': f'A_Pre_W{i+1}',
                        'Area-ZSB': f'A_Post_W{i+1}',
                        'Area-FS': f'A_Full_W{i+1}',
                        'Area-Perc-KYY': f'A%_Pre_W{i+1}',
                        'Area-Perc-ZSB': f'A%_Post_W{i+1}',
                        'Area-Perc-FS': f'A%_Full_W{i+1}'}, axis=1).copy()

        if df_all.empty:
            df_all = tempdf.copy()
        else:
            tempdf.drop(columns=['GroupID'], inplace=True)
            df_all = pd.merge(df_all, tempdf, on="ParticipantID", how="left")

    #* df with areas for the full DS for each metric weight
    df_FullArea = pd.DataFrame(DSAreas_lst, columns=['FullDSArea'])

    #! get column of interest (Full area) only
    df_AreaWeight_Plot = pd.DataFrame()
    idx_list = []

    for i in range(len(dfsAreas_lst)):
        idx_list.append(f'A%_Full_W{i+1}')

        if i == 0:
            df_AreaWeight_Plot = df_all[['ParticipantID', f'A%_Full_W{i+1}']].copy().T
        else:
            df_AreaWeight_Plot = pd.concat([df_AreaWeight_Plot, df_all[[f'A%_Full_W{i+1}']].copy().T], ignore_index=True)

    df_AreaWeight_Plot.columns = df_AreaWeight_Plot.iloc[0]
    df_AreaWeight_Plot.drop(df_AreaWeight_Plot.index[0], inplace=True)

    #* stack all columns in single column (transpose, then stack)
    df_AreaWeight_Plot = df_AreaWeight_Plot.T.stack().reset_index(drop=True)

    #!reconstruct df ( adding participant number and weight tag)
    ids_list = df_all['ParticipantID'].unique()

    df_fin = pd.DataFrame(df_AreaWeight_Plot, columns=['values'])
    df_fin['participant'] = ''
    df_fin['weight'] = ''

    #* add weight type
    for i in np.arange(0, len(df_fin), step=11):
        for j in range(11):
            df_fin.loc[j+i:j+1+i,'weight'] = idx_list[j]

    #* add participant
    for i in np.arange(0, len(df_fin), step=1):
        for j in range(11):
            df_fin.loc[j+i:j+1+i,'participant'] = ids_list[i//11]

    dir_val = Path(f'{dir_data.parent}'+r'/validation')

    #! lineplot
    fig = px.line(df_fin, x='weight', y='values', color='participant', markers=True)
    fig.show()
    if save_plt == True:
        fig.write_html(f'{dir_val}/sensitivity_area_validation.html')

    #! boxplot
    fig = px.box(df_fin, x='participant', y="values", color='participant')
    fig.show()
    if save_plt == True:
        fig.write_html(f'{dir_val}/sensitivity_area_validation_boxplot.html')

    #!save metrics to excel
    with pd.ExcelWriter(f'{dir_val}/DS_AreaSensitivityAnalysis.xlsx') as writer:
        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        df_all.to_excel(writer, sheet_name="AllWeights", index=False)
        df_FullArea.to_excel(writer, sheet_name="FullDSAreas", index=False)

