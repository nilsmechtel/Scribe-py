import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def vis_causal_net(adata, layout='circular', top_n_edges=10, edge_color='gray', width='weight', node_size=100, node_color='skyblue',
                   figsize =(6,6), figure=None):
    """Visualize inferred causal regulatory network

    This plotting function visualize the inferred causal regulatory network inferred from Scribe.

    Arguments
    ---------
        adata: `Anndata`
            Annotated Data Frame, an Anndata object.
        layout: `str` (Default: circular)
            A string determines the graph layout function supported by networkx. Currently supported layouts include
            circular, kamada_kawai, planar, random, spectral, spring and shell.
        top_n_edges: 'int' (default 10)
            Number of top strongest causal regulation to visualize.
        edge_color: `str` (Default: gray)
            The color for the graph edge.
        figsize: `tuple` (Default: (6, 6))
            The tuple of the figure width and height.

    Returns
    -------
        A figure created by nx.draw and matplotlib.
    """

    if 'causal_net' not in adata.uns.keys():
        raise('Causal_net is not a key in uns slot. Please first run causal network inference with Scribe.')

    df_mat = deepcopy(adata.uns['causal_net'])

    if df_mat.shape[0] == df_mat.shape[1]:
        tmp = np.where(df_mat.values - df_mat.T.values < 0)
        for i in range(len(tmp[0])):
            df_mat.iloc[tmp[0][i], tmp[1][i]] = np.nan

    df_mat = df_mat.stack().reset_index()
    df_mat.columns = ['source', 'target', 'weight']

    if top_n_edges is not None and top_n_edges < len(df_mat):
        df_mat = df_mat.sort_values(by='weight', ascending=False)[:top_n_edges]

    G = nx.from_pandas_edgelist(df_mat, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())
    G.nodes()
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr['weight'])

    if type(layout) is not str:
        pos = layout
    elif layout is None:
        pos = None
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "planar":
        pos = nx.planar_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "spiral":
        pos = nx.spiral_layout(G)
    else:
        raise('Layout', layout, ' is not supported.')

    if edge_color == 'weight':
        edge_color = W 

    if width == 'weight':
        width = W * 15

    if figure is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figure[0], figure[1]

    options = {
    'width': 300,
    'arrowstyle': '-|>',
    'arrowsize': 1000,
     }
     
    nx.draw(G, ax=ax,
            pos=pos,
            with_labels=True,
            node_color=node_color, node_size=node_size,
            edge_color=edge_color, width=width,
            edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=1, 
            options=options) 

    if figure is None:
        plt.show()
    else:
        return pos, fig

