import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from copy import deepcopy

def vis_causal_net(adata, source_genes = None, target_genes = None, layout = 'circular', top_n_edges = 10,
                   edge_color = 'gray', width='weight', min_W = None, max_W = None, node_color = 'skyblue', node_size = 100, figsize = (6, 6),
                   save_to = None, show = True, fig = None, ax = None, face_color='w'):
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

    all_genes = list(set(df_mat['source'].tolist() + df_mat['target'].tolist()))
    def check_genes(input_list, s_t):
        output_list = []
        for gene in input_list:
            if gene not in all_genes:
                print('\n', '%s has been removed from %s genes.' % (gene, s_t))
            else:
                output_list.append(gene)
        return output_list

    if source_genes is not None:
        source_genes = check_genes(source_genes, 'source')
        df_mat = df_mat[[gene in source_genes for gene in df_mat['source']]]

    if target_genes is not None:
        target_genes = check_genes(target_genes, 'target')
        df_mat = df_mat[[gene in target_genes for gene in df_mat['target']]]

    if top_n_edges is not None and top_n_edges < len(df_mat):
        df_mat = df_mat.sort_values(by='weight', ascending=False)[:top_n_edges]

    source_genes = df_mat['source'].tolist()
    target_genes = df_mat['target'].tolist()

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
    else:
        raise('Layout', layout, ' is not supported.')

    if min_W is None:
        min_W = np.min(W)

    if max_W is None:
        max_W = np.max(W)

    if edge_color == 'weight':
        edge_color = W 

    if width == 'weight':
        width = W / max_W * 8

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
     
    nx.draw(G, ax=ax,
            pos=pos,
            with_labels=True,
            node_color=node_color, node_size=node_size,
            edge_color=edge_color, width=width,
            edge_cmap=plt.cm.Blues, edge_vmin=min_W, edge_vmax=max_W, 
            options = options) 

    if save_to is not None:
        if path is None:
            path = os.getcwd()
        else:
            tmp_path = ''
            splitted_path = path.split('/')
            for i, folder in enumerate(splitted_path):
                if i == 0: continue
                tmp_path += '/%s' % splitted_path[i]
                if not os.path.isdir(tmp_path):
                    try:
                        os.mkdir(tmp_path)
                    except OSError:
                        print('\n', "Creation of the directory %s failed" % tmp_path)
                    else:
                        print('\n', "Successfully created the directory %s" % tmp_path)

        plt.ioff()
        plt.savefig(save_to, dpi=None, facecolor=face_color, edgecolor=face_color,
                    orientation='portrait', papertype=None, format='png',
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)
    
    if show is True:
        plt.show()

    return (source_genes, target_genes, pos, (min_W,max_W), fig)

