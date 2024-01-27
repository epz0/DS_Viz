"""The dim_reduction module creates a 2D embedding of a given distance matrix (using UMAP) and plots the points in a 2D space.

Functions:
    create_embedding: Function that creates the embedding of the distance matrix.
    plot_embedding: Function to plot the embedding as points, colored by participant.

Obs: currently only UMAP is being used for creating the embedding.
"""

from pathlib import Path
from matplotlib import pyplot as plt

import igraph
import numpy as np
import pandas as pd
import umap


def create_embedding(dir_data, dmatrix, embed_name=None, Wg='W2', NN=60, MD=0.25, densm=False):
    """Returns embedding of distance matrix and its graph object.

    Args:
        dir_data (path): Path to the directory with the data file.
        dmatrix (NPArray): Normalised, pairwise distance matrix.
        embed_name (string, optional): Name of the embedding that should be created. Defaults to None.
        NN (int, optional): Number of neighbors parameter for the UMAP embedding. Defaults to 120 (from validation process).
        MD (float, optional): Number of minimum distance parameter for the UMAP embedding. Defaults to 0.3 (from validation process).
        densm (float, optional): Defaults to false. Expectes float to be passed onto the dens_lambda parameter.

    Returns:
        embedding_umap: Matrix with the embedding (x,y points) calculated from UMAP.
        DS_graph: Graph object of the embedding.
    """

    dir_exp = Path(f'{dir_data.parent}/export')

    if embed_name is None:
        embed_name = f'{Wg}_{NN}_{MD}_dM{densm}'
    else:
        embed_name = f'{embed_name}_{NN}_{MD}_dM{densm}'

    # check if graph/embedding already exists
    # Read graph if it exists
    my_graph = Path(f'{dir_exp}/DS_{embed_name}.pkl')

    if my_graph.is_file():
        print('UMAP exists - reading from file!')
        graph = pd.read_pickle(my_graph)
        embed = pd.read_csv(Path(f'{dir_exp}/DS_{embed_name}.csv'))
        embedding_umap = embed[['x','y']].to_numpy()

        # Construct igraph object
        DS_graph = igraph.Graph.TupleList(
                                    graph.itertuples(index=False),
                                    directed = False,
                                    edge_attrs= ['weight']
                                    )

    else:
        print(f'Calculating new UMAP, densMAP={densm}...')
        if densm == False:
            # create new embedding/graph
            #*set dim redux parameters [VIZ]
            reducer = umap.UMAP(n_components = 2,
                                metric='precomputed',       #* metric precomputed (square matrix)
                                min_dist=MD,                #* dist between embedded points, higher less overlap between points default=0.1
                                n_neighbors=NN,             #* larger --> more global usually between 5-100
                                random_state=143,           # for reproduceability
                                #repulsion_strength=1,      #* default
                                #n_epochs=1000,             #* 500 for small datasets
                                #spread=1,                  #* default
                                #init='spectral',
                                n_jobs=-1)

            embedding_umap = reducer.fit_transform(dmatrix) #! return

            #* export umap graph
            df_embed = pd.DataFrame(embedding_umap, columns=['x','y'])

            coo_graph = reducer.graph_.tocoo()
            graph = pd.DataFrame(np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
                                    columns=('source', 'target', 'weight'))
            graph.to_pickle(f'{dir_exp}/DS_{embed_name}.pkl')

            #* assign graph
            DS_graph = igraph.Graph.TupleList(
                                        graph.itertuples(index=False),
                                        directed = False,
                                        edge_attrs= ['weight']
                                        )                   #! return

            #! export umap embedding df as csv
            df_embed.to_csv(f'{dir_exp}/DS_{embed_name}.csv')

        else:
            #** densmap = true
            # create new embedding/graph
            #*set dim redux parameters [VIZ]
            reducer = umap.UMAP(densmap=True,
                                dens_lambda=densm,
                                n_components = 2,
                                metric='precomputed',       #* metric precomputed (square matrix)
                                min_dist=MD,                #* dist between embedded points, higher less overlap between points default=0.1
                                n_neighbors=NN,             #* larger --> more global usually between 5-100
                                random_state=143,           # for reproduceability
                                #repulsion_strength=1,      #* default
                                #n_epochs=1000,             #* 500 for small datasets
                                #spread=1,                  #* default
                                #init='spectral',
                                n_jobs=-1)

            embedding_umap = reducer.fit_transform(dmatrix) #! return

            #* export umap graph
            df_embed = pd.DataFrame(embedding_umap, columns=['x','y'])

            coo_graph = reducer.graph_.tocoo()
            graph = pd.DataFrame(np.vstack([coo_graph.row, coo_graph.col, coo_graph.data]).T,
                                    columns=('source', 'target', 'weight'))
            graph.to_pickle(f'{dir_exp}/DS_{embed_name}.pkl')

            #* assign graph
            DS_graph = igraph.Graph.TupleList(
                                        graph.itertuples(index=False),
                                        directed = False,
                                        edge_attrs= ['weight']
                                        )                   #! return

            #! export umap embedding df as csv
            df_embed.to_csv(f'{dir_exp}/DS_{embed_name}.csv')
        print('UMAP embedding generated!')
    return embedding_umap, DS_graph


def plot_embedding(dir_data, embed, df, embed_name='60_0.25'):
    """Plots the embeeding generated and saves the image as .png.

    Args:
        dir_data (path):  Path to the directory with the data file.
        embed (matrix): Full embedding with x,y points coordinates.
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        embed_name (string): Name of the embedding Defaults to 120_0.3 (default values of NN and MD).
    """

    #* directory for the output viz
    dir_viz = Path(f'{dir_data.parent}'+r'/viz/DS_points')
    print('Plotting embedding...')

    #* plotting UMAP Embedding original (just points)
    plt.rcParams["figure.figsize"] = (10,10)
    fig, ax = plt.subplots()
    scatter = ax.scatter(embed[:, 0],embed[:, 1], c=df['colors'], s=10)
    plt.title(f'DS Viz | {embed_name}', fontsize=16, fontweight='bold')
    plt.savefig(f'{dir_viz}/DS_{embed_name}.png', dpi=300, bbox_inches='tight')

    print('Plotting done!')



