"""The dist_matrix module creates distance matrices. Two possibilities exist: a) calculating a distance metric from the data or b) calculating the distance matrix from a set of 2D points.

Functions:
    calc_distmatrix: Main function that creates a normalised distance matrix given raw analysis data and metric parameters.
    jac_dist: Function to calculate the Jaccard distance between two boolean lists (features), feeding into the matrix calculation.
    create_dmatrix_from_embed: Function to create a normalised distance matrix from a set of 2D points (e.g., from UMAP embedding).
"""

import ast
import math
import gower
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import combinations
from scipy.spatial import  distance


def jac_dist(l1, l2):
    """Returns the Jaccard distance between two lists.

    Args:
        l1 (Bool): List of boolean values for the solution {i} characteristics.
        l2 (Bool): List of boolean values for the solution {i+1} characteristics.

    Returns:
        distance (float): Jaccard distance between the two solutions.
    """
    list1 = ast.literal_eval(l1)
    list2 = ast.literal_eval(l2)
    return distance.jaccard(list1,list2)

def calc_distmatrix(df, dir_data, fname, sheetjac=None, sheetgow=None, jacweight=None, gowerweight=None, mult_gow=None):
    """Returns and saves the normalised distance matrix from the combined Gower and Jaccard distances.

    Args:
        df (dataframe): Main dataframe (df_base) with all info from the spreadsheet.
        dir_data (path): Path to the directory with the data file.
        fname (string): Name of the file containing the data for the metrics' weights. Expects Excel file.
        sheetjac (string): Sheet name with data for the Jaccard distance weights. Defaults to None.
        sheetgow (string): Sheet name with data for the Gower distance weights. Defaults to None.
        jacweight (float, optional): Weight of the Jaccard distance calculated in the combined metric. Defaults to None, which takes the max value of the Gower weights.
        gowerweight (list, optional): List with values for the Gower distance calculation. Defaults to None, taking the specified weighting.
        mult_gow (integer, optional): Index to get name for file when running multiple calculations of the gower distance.

    Returns:
        n_distmatrix: Normalised matrix with pairwise distances (distance metric = combination of Jaccard and Gower)
    """

    dir_exp = Path(f'{dir_data.parent}/export')

    print("Getting metrics' weights...")
    #* define weights for the Gower/Jaccard metrics
    if sheetjac is None:
        sheetjac = 'Jaccard_Weight'

    if sheetgow is None:
        sheetgow = 'Gower_Weight'

    df_gowerw = pd.read_excel(f'{dir_data}/{fname}', sheet_name=sheetgow)
    df_jacw = pd.read_excel(f'{dir_data}/{fname}', sheet_name=sheetjac)             

    # Getting Gower dist with weights from Excel cell
    if gowerweight is None:
        gwr_wgt = ast.literal_eval(df_gowerw['gowerW'][0])                          #! specified Gower weighting
        gw = f'G_{df_gowerw["WTp"][0]}'
    else:
        gw = f'G_{df_gowerw["WTp"][mult_gow]}'                                      #* expects list of size equals to the number of Gower features

    #* weight of the Jaccard distance in the final calc
    if jacweight is None:
        weight_jac = df_jacw['w_jac'][0]                         #! default Jaccard weight from spreadsheet
        jw = f'J_W{weight_jac}'
    else:
        weight_jac = jacweight                                  #* expects Jaccard weight
        jw = f'J_W{weight_jac}'

    #* check if dist matrix exists and gets it from file
    d_matrix = Path(f'{dir_exp}/d_matrix_{gw}_{jw}.npy')

    if d_matrix.is_file():
        print('Distance matrix exists - reading from file!')

        with open(d_matrix, 'rb') as f:
            n_distmatrix = np.load(f)
        f.close

    else:
        print('Calculating distance matrix...')
        #* subset the df to get only info needed for gower distance calculation
        df_gow = df[['FullID','type', 'budgetUsed',
            'numAnchorsUsed', 'deckType_1', 'deckType_2',
            'structurePosition_Top', 'structurePosition_Bottom',
            'structurePosition_Rock', 'roadAsStrut', 'rockSupportShape', 'rockSupportMat',
            'structureSize', 'materialRoad', 'materialReinfRoad', 'materialWood',
            'materialSteel', 'MaterialOther', 'leftAnchorConnected', 'rightAnchorConnected',
            'midAnchorConnected', 'hasDangling', 'restOnRock',
            'predominantMaterialSupport', 'predominantMaterialDeck', 'hasDiamonds',
            'hasStructure', 'doubleFeature', 'isModded', 'exploitType']]

        print('Calculating Jaccard distance matrix...')
        #* Jaccard distance calc
        # function gets 2 lists from excel cell and returns the Jaccard distance between them
        # creating the m x m matrix to receive all the pairwise distances
        dmx_jac = np.zeros((len(df), len(df)))

        # distances calculated calling the function, only for the upper diagonal (simmetric matrix)
        dmx_jac[np.triu_indices(len(df), k=1)] = [jac_dist(a,b) for a,b in combinations(df['listJac'], r=2)]

        # filling out the matrix with the transposed
        # diagonal = 0 --> d(a,a) = 0
        dmx_jac += dmx_jac.T
        np.fill_diagonal(dmx_jac,0)
        print('Jaccard distance matrix done!')

        print('Calculating Gower distance matrix...')


        #* Calculating the Gower distance matrix
        dmx_gower = gower.gower_matrix(df_gow.drop(columns=['FullID']))
        print('Gower distance matrix done!')

        print('Calculating combined, normalised distance matrix...')


        dmx_jac = dmx_jac * weight_jac

        #* Final distmetric calculation (Jaccard + Gower dist)
        distmatrix = dmx_gower + dmx_jac                            # simmetric, diag = 0

        #! normalised dist matrix (the one to be used)
        n_distmatrix = np.transpose((distmatrix-np.min(distmatrix,axis=1))/(np.max(distmatrix)-np.min(distmatrix,axis=1)))

        #* save matrix to .npy (numpy file)
        with open(f'{dir_exp}/d_matrix_{gw}_{jw}.npy', 'wb') as f:
            np.save(f, n_distmatrix)
        f.close

        print('Normalised distance matrix done!')

    return n_distmatrix

def create_dmatrix_from_embed(dir_data, embed=None, embed_name=None, NN=None, MD=None, Dens=None, norm=True):
    """Returns a normalised distance matrix from a 2D embedding.

    Args:
        dir_data (path): Path to the directory with the data file.
        embed (matrix, optional): Matrix with the embedding (x,y points) calculated from UMAP. Defaults to None. If not passed, tries to read from file.
        embed_name (string, optional): Name of the embedding that should be loaded. Defaults to None.

    Raises:
        ValueError: No embedding found/passed, raised when no embed is passed and none is found in the folder.

    Returns:
        n_dmx_umap: Normalised distance matrix calculated from the 2D embedding.
    """

    dir_exp = Path(f'{dir_data.parent}/export')

    if embed is None:
        if embed_name is None:
            embed_name = 'DS_50_0.3'                                   # default values of NN and MD
            print (f'Using default embed configuration/name {embed_name}...')
        else:
            if NN is not None and MD is not None and Dens is not None:
                embed_name = f'DS_{embed_name}_{NN}_{MD}_dM{Dens}'
            elif NN is not None and MD is not None and Dens is None:
                embed_name = f'DS_{embed_name}_{NN}_{MD}'

        # check if graph/embedding already exists
        # Read graph if it exists
        my_graph = Path(f'{dir_exp}/DS_{embed_name}.pkl')

        if my_graph.is_file():
            print('Embedding exists - reading from file...')
            graph = pd.read_pickle(my_graph)
            embed = pd.read_csv(Path(f'{dir_exp}/DS_{embed_name}.csv'))
            embedding_umap = embed[['x','y']].to_numpy()
            print (f'Embedding read!')
        else:
            raise ValueError("No embedding found/passed!")
    else:
        embedding_umap = embed

    print('Calculating distance matrix from 2D embedding points...')
    #! calculating distmatrix from umap points
    all_x_points = embedding_umap[:,0]
    all_y_points = embedding_umap[:,1]
    x_y_points = list(zip(all_x_points,all_y_points)) #list of points

    # creating the m x m matrix to receive all the pairwise distances
    dmx_umap = np.zeros((len(embedding_umap), len(embedding_umap)))

    # distances calculated calling the function, only for the upper diagonal (simmetric matrix)
    dmx_umap[np.triu_indices(len(embedding_umap), k=1)] = [math.dist(a,b) for a,b in combinations(x_y_points, r=2)]

    # filling out the matrix with the transposed
    # diagonal = 0 --> d(a,a) = 0
    dmx_umap += dmx_umap.T
    np.fill_diagonal(dmx_umap,0)

    if norm == True:
        # normalising dmx_umap
        n_dmx_umap = np.transpose((dmx_umap-np.min(dmx_umap,axis=1))/(np.max(dmx_umap)-np.min(dmx_umap,axis=1)))
        return n_dmx_umap
    elif norm == False:
        return dmx_umap

    print('Calculating distance matrix from 2D embedding points done!')
