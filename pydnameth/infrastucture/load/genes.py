from pydnameth.infrastucture.path import get_cache_path
import numpy as np
import os.path
import pickle
from pydnameth.infrastucture.load.betas import load_betas
from pydnameth.infrastucture.load.betas_adj import load_betas_adj
from pydnameth.infrastucture.load.residuals import load_residuals_common
from tqdm import tqdm


def load_genes(config):

    suffix_gene = ''
    if bool(config.experiment.data_params):
        suffix_gene += '_' + str(config.experiment.get_data_params_str())
        source = config.experiment.data_params.pop('source')
    else:
        raise ValueError(f'Data params for genes are empty')

    fn_list_txt = get_cache_path(config) + '/' + 'genes_list.txt'
    fn_list_pkl = get_cache_path(config) + '/' + 'genes_list.pkl'
    fn_dict_pkl = get_cache_path(config) + '/' + 'genes_dict.pkl'
    fn_missed_dict_pkl = get_cache_path(config) + '/' + 'genes_missed_dict.pkl'
    fn_data_npz = get_cache_path(config) + '/' + 'genes' + suffix_gene + '.npz'
    fn_data_txt = get_cache_path(config) + '/' + 'genes' + suffix_gene + '.txt'

    if os.path.isfile(fn_dict_pkl) and os.path.isfile(fn_list_pkl) and os.path.isfile(fn_data_npz):

        f = open(fn_list_pkl, 'rb')
        config.genes_list = pickle.load(f)
        f.close()

        f = open(fn_dict_pkl, 'rb')
        config.genes_dict = pickle.load(f)
        f.close()

        f = open(fn_missed_dict_pkl, 'rb')
        config.genes_missed_dict = pickle.load(f)
        f.close()

        data = np.load(fn_data_npz)
        config.genes_data = data['data']

    else:

        if source == 'betas':
            load_betas(config)
            source_dict = config.betas_dict
            source_data = config.betas_data
            source_missed_dict = config.betas_missed_dict
        elif source == 'betas_adj':
            load_betas_adj(config)
            source_dict = config.betas_adj_dict
            source_data = config.betas_adj_data
            source_missed_dict = config.betas_adj_missed_dict
        elif source == 'residuals':
            load_residuals_common(config)
            source_dict = config.residuals_dict
            source_data = config.residuals_data
            source_missed_dict = config.residuals_missed_dict
        else:
            raise ValueError(f'Source for genes is not specified')

        num_subjects = config.betas_data.shape[1]

        config.genes_list = []
        for gene_id, gene in tqdm(enumerate(config.gene_cpg_dict), mininterval=60.0, desc='genes_list creating'):
            cpgs = config.gene_cpg_dict[gene]
            for cpg in cpgs:
                if cpg in source_dict:
                    config.genes_list.append(gene)
                    break

        config.genes_dict = {}
        config.genes_missed_dict = {'any': []}
        config.genes_data = np.zeros((len(config.genes_list), num_subjects), dtype=np.float32)

        for gene_id, gene in tqdm(enumerate(config.genes_list), mininterval=60.0, desc='genes_data creating'):
            config.genes_dict[gene] = gene_id

            cpgs = config.gene_cpg_dict[gene]

            denominators = np.zeros(num_subjects, dtype=np.float32)

            for cpg in cpgs:
                if cpg in source_dict:
                    row_id = source_dict[cpg]
                    source_values_raw = source_data[row_id, :]
                    if len(source_missed_dict[cpg]) > 0:
                        source_values = np.zeros(num_subjects, dtype=np.float32)
                        for val_id in range(0, num_subjects):
                            if val_id not in source_missed_dict[cpg]:
                                source_values[val_id] = source_values_raw[val_id]
                                denominators[val_id] += 1.0

                    else:
                        source_values = source_values_raw
                        denominators += np.ones(num_subjects, dtype=np.float32)
                    config.genes_data[gene_id] += source_values

            for val_id in range(0, num_subjects):
                config.genes_data[gene_id][val_id] /= denominators[val_id]

        f = open(fn_list_pkl, 'wb')
        pickle.dump(config.genes_list, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        f = open(fn_dict_pkl, 'wb')
        pickle.dump(config.genes_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        f = open(fn_missed_dict_pkl, 'wb')
        pickle.dump(config.genes_missed_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        np.savez_compressed(fn_data_npz, data=config.genes_data)
        np.savetxt(fn_data_txt, config.genes_data, delimiter='\t', fmt='%.8e')

        with open(fn_list_txt, 'w') as f:
            for item in config.genes_list:
                f.write("%s\n" % item)
