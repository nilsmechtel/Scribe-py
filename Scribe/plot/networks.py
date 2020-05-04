import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def vis_causal_net(adata, layout = 'circular', top_n_edges = 10, edge_color = 'gray', figsize=(6, 6), save=None):
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
        raise('causal_net is not a key in uns slot. Please first run causal network inference with Scribe.')

    df_mat = adata.uns['causal_net']
    ind_mat = np.where(df_mat.values - df_mat.T.values < 0)

    tmp = np.where(df_mat.values - df_mat.T.values < 0)

    for i in range(len(tmp[0])):
        df_mat.iloc[tmp[0][i], tmp[1][i]] = np.nan

    df_mat = df_mat.stack().reset_index()
    df_mat.columns = ['source', 'target', 'weight']

    if top_n_edges is not None:
        ind_vec = np.argsort(-df_mat.loc[:, 'weight'])
        df_mat = df_mat.loc[ind_vec[:top_n_edges], :]

    G = nx.from_pandas_edgelist(df_mat, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())
    G.nodes()
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr['weight'])

    options = {
    'width': 300,
    'arrowstyle': '-|>',
    'arrowsize': 1000,
     }

    plt.figure(figsize=figsize)
    if layout is None:
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    elif layout is "circular":
        nx.draw_circular(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    elif layout is "kamada_kawai":
        nx.draw_kamada_kawai(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    elif layout is "planar":
        nx.draw_planar(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    elif layout is "random":
        nx.draw_random(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    elif layout is "spectral":
        nx.draw_spectral(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    elif layout is "spring":
        nx.draw_spring(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    elif layout is "shell":
        nx.draw_shell(G, with_labels=True, node_color='skyblue', node_size=100, edge_color=edge_color, width=W / np.max(W) * 5, edge_cmap=plt.cm.Blues, options = options)
    else:
        raise('layout', layout, ' is not supported.')
    if save is not None:
        plt.savefig(save, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
    plt.show()


def vis_causal_net2(adata, top_n_edges = 10, node_color='skyblue', figsize=(6, 6), save=None):

    if 'causal_net' not in adata.uns.keys():
        raise('causal_net is not a key in uns slot. Please first run causal network inference with Scribe.')

    df_mat = adata.uns['causal_net']
    ind_mat = np.where(df_mat.values - df_mat.T.values < 0)

    tmp = np.where(df_mat.values - df_mat.T.values < 0)

    for i in range(len(tmp[0])):
        df_mat.iloc[tmp[0][i], tmp[1][i]] = np.nan

    df_mat = df_mat.stack().reset_index()
    df_mat.columns = ['source', 'target', 'weight']

    adata.var['gene_name'] = adata.var_names.tolist()

    df_mat.source = adata.var.loc[df_mat.source, 'gene_name'].tolist()
    df_mat.target = adata.var.loc[df_mat.target, 'gene_name'].tolist()

    df_mat = df_mat.iloc[~np.isinf(df_mat.weight.tolist()), :]

    top_edges = df_mat.weight >= np.quantile(df_mat.weight, 1 - top_n_edges / len(df_mat))
    df_mat = df_mat.loc[top_edges,]

    G = nx.from_pandas_edgelist(df_mat, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph())
    G.nodes()
    W = []
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            W.append(eattr['weight'])

    G.nodes()
    plt.figure(figsize=figsize)
    nx.draw(G, with_labels=True, node_color=node_color, node_size=100, edge_color=W, width=1.0, edge_cmap=plt.cm.Blues)
    if save is not None:
        plt.savefig(save, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
    plt.show()
