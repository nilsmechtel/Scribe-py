import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix

from .causal_network import cmi
from multiprocessing import Pool

tmp_input = []
def pool_cmi(pos):
    gene_a = tmp_input[pos][0]
    gene_b = tmp_input[pos][1]
    x = tmp_input[pos][2]
    y = tmp_input[pos][3]
    z = tmp_input[pos][4]
    return (gene_a, gene_b, cmi(x, y, z))

def causal_net_dynamics_coupling(adata, regulator_genes=None, target_genes=None, guide_keys=None, t0_key='spliced', t1_key='velocity', normalize=True, copy=False, number_of_processes=1):
    """Infer causal networks with dynamics-coupled single cells measurements.
    Network inference is a insanely challenging problem which has a long history and that none of the existing algorithms work well.
    However, it's quite possible that one or more of the algorithms could work if only they were given enough data. Single-cell
    RNA-seq is exciting because it provides a ton of data. Somewhat surprisingly, just having a lot of single-cell RNA-seq data
    won't make causal inference work well. We need a fundamentally better type of measurement that couples information across
    cells and across time points. Experimental improvements are coming now, and whether they are sufficient to power methods
    like Scribe is important future work. For example, the recent developed computational algorithm (La Manno et al. 2018) estimates
    the levels of new (unspliced) versus mature (spliced) transcripts from single-cell RNA-seq data for free. Moreover, exciting
    experimental approaches, like single cell SLAM-seq methods (Hendriks et al. 2018; Erhard et al. 2019; Cao, Zhou, et al. 2019)
    are recently developed that measures the transcriptome of two time points of the same cells. Datasets generated from those methods
    will provide improvements of causal network inference as we comprehensively demonstrated from the manuscript. This function take
    advantages of those datasets to infer the causal networks.

    We note that those technological advance may be still not sufficient, radically different methods, for example something like
    highly multiplexed live imaging that can record many genes may be needed.

    Arguments
    ---------
    adata: `anndata`
        Annotated data matrix.
    regulator_genes: `List`
    target_genes: `List`
    guide_keys: `List` (default: None)
        Whether to scale the expression or velocity values into 0 to 1 before calculating causal networks.
    t0_key: `str` (default: spliced)
        Key corresponds to the transcriptome of the initial time point, for example spliced RNAs from RNA velocity, old RNA
        from scSLAM-seq data.
    t1_key: `str` (default: velocity)
        Key corresponds to the transcriptome of the next time point, for example unspliced RNAs (or estimated velocitym,
        see Fig 6 of the Scribe preprint) from RNA velocity, old RNA from scSLAM-seq data.
    normalize: `bool`
        Whether to scale the expression or velocity values into 0 to 1 before calculating causal networks.
    copy: `bool`
        Whether to return a copy of the adata or just update adata in place.

    Returns
    ---------
    An update AnnData object with inferred causal network stored as a matrix related to the key `causal_net` in the `uns` slot.
    """

    if regulator_genes is None:
        regulator_genes = adata.var_names.values.tolist()
    if target_genes is None:
        target_genes = adata.var_names.values.tolist()

    genes = list(set(regulator_genes + target_genes))

    if guide_keys is not None:
        genes = np.unique(adata.obs[guide_keys].tolist())
        genes = np.setdiff1d(genes, ['*', 'nan', 'neg'])

        idx_var = [vn in genes for vn in adata.var_names]
        idx_var = np.argwhere(idx_var)
        genes = adata.var_names.values[idx_var.flatten()].tolist() #[idx_var]
        regulator_genes = genes
        target_genes = genes

    # support sparse matrix:
    tmp = pd.DataFrame(adata.layers[t0_key].todense()) if isspmatrix(adata.layers[t0_key]) else pd.DataFrame(adata.layers[t0_key])
    tmp.index = adata.obs_names
    tmp.columns = adata.var_names
    spliced = tmp.loc[:, genes]
    tmp = pd.DataFrame(adata.layers[t1_key].todense()) if isspmatrix(adata.layers[t1_key]) else pd.DataFrame(adata.layers[t1_key])
    tmp.index = adata.obs_names
    tmp.columns = adata.var_names
    velocity = tmp.loc[:, genes]
    velocity[pd.isna(velocity)] = 0 # set NaN value to 0

    if normalize:
        spliced = (spliced - spliced.mean()) / (spliced.max() - spliced.min())
        velocity = (velocity - velocity.mean()) / (velocity.max() - velocity.min())

    causal_net = pd.DataFrame({node_id: [np.nan for i in regulator_genes] for node_id in target_genes}, index=regulator_genes)

    for g_a in regulator_genes:
        for g_b in target_genes:
            if g_a == g_b:
                continue
            else:
                x_orig = spliced.loc[:, g_a].tolist()
                y_orig = (spliced.loc[:, g_b] + velocity.loc[:, g_b]).tolist() if t1_key is 'velocity' else spliced.loc[:, g_b].tolist()
                z_orig = velocity.loc[:, g_b].tolist()

                # input to cmi is a list of list
                x_orig = [[i] for i in x_orig]
                y_orig = [[i] for i in y_orig]
                z_orig = [[i] for i in z_orig]

                if number_of_processes == 1:
                    causal_net.loc[g_a, g_b] = cmi(x_orig, z_orig, y_orig)
                else:
                    tmp_input.append([g_a, g_b, x_orig, z_orig, y_orig])

    if number_of_processes > 1:
        pool = Pool(number_of_processes)
        tmp_results = pool.map(pool_cmi, [pos for pos in range(len(tmp_input))])
        pool.close()
        pool.join()
        for t in tmp_results:
            causal_net.loc[t[0], t[1]] = t[2]
        
    adata.uns['causal_net'] = causal_net

#     logg.info('     done', time = True, end = ' ' if settings.verbosity > 2 else '\n')
#     logg.hint('perturbation_causal_net \n'
#               '     matrix is added to adata.uns')

    return adata if copy else None
