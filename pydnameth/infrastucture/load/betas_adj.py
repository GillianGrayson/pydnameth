from pydnameth.infrastucture.load.betas import load_betas
from pydnameth.infrastucture.load.residuals import load_residuals
from pydnameth.infrastucture.path import get_data_base_path
import numpy as np
import pickle
import os.path
from tqdm import tqdm


def load_betas_adj(config):

    suffix = ''
    if bool(config.experiment.data_params):
        suffix += '_' + config.experiment.get_data_params_str()
    else:
        raise ValueError(f'Exog for residuals is empty.')

    fn_dict = get_data_base_path(config) + '/' + 'betas_dict' + suffix + '.pkl'
    fn_missed_dict = get_data_base_path(config) + '/' + 'betas_missed_dict' + suffix + '.pkl'
    fn_data = get_data_base_path(config) + '/' + 'betas_adj' + suffix + '.npz'

    if os.path.isfile(fn_dict) and os.path.isfile(fn_data):

        f = open(fn_dict, 'rb')
        config.betas_adj_dict = pickle.load(f)
        f.close()

        f = open(fn_missed_dict, 'rb')
        config.betas_adj_missed_dict = pickle.load(f)
        f.close()

        data = np.load(fn_data)
        config.betas_adj_data = data['data']

    else:
        load_residuals(config)

        config.experiment.data_params = {}
        load_betas(config)

        config.betas_adj_dict = config.residuals_dict
        f = open(fn_dict, 'wb')
        pickle.dump(config.betas_adj_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        config.betas_adj_missed_dict = config.residuals_missed_dict
        f = open(fn_missed_dict, 'wb')
        pickle.dump(config.betas_missed_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        num_cpgs = config.betas_data.shape[0]
        num_subjects = config.betas_data.shape[1]
        config.betas_adj_data = np.zeros((num_cpgs, num_subjects), dtype=np.float32)

        for cpg in tqdm(config.betas_adj_dict, mininterval=60.0, desc='betas_adj_data creating'):

            residuals = config.residuals_data[config.residuals_dict[cpg], :]
            betas = config.betas_data[config.betas_dict[cpg], :]

            min_residuals = np.min(residuals)
            mean_betas = np.mean(betas)

            shift = mean_betas
            if min_residuals + shift < 0:
                shift = abs(min_residuals)

            betas_adj = residuals + shift

            config.betas_adj_data[config.residuals_dict[cpg]] = betas_adj

        np.savez_compressed(fn_data, data=config.betas_adj_data)

        # Clear data
        del config.residuals_data
        del config.betas_data
