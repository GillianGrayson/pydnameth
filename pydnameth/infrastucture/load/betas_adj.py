from pydnameth.infrastucture.load.betas import load_betas
from pydnameth.infrastucture.path import get_data_base_path
from pydnameth.infrastucture.load.attributes import load_cells_dict, load_observables_dict
import numpy as np
import pandas as pd
from statsmodels import api as sm
import pickle
import os.path
from tqdm import tqdm
import copy


def load_betas_adj(config):

    suffix = ''
    if bool(config.experiment.data_params):
        data_params = config.experiment.data_params
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
        config.betas_missed_dict = pickle.load(f)
        f.close()

        data = np.load(fn_data)
        config.betas_adj_data = data['data']

    else:

        config.experiment.data_params = {}
        load_betas(config)

        config.betas_adj_dict = config.betas_dict
        f = open(fn_dict, 'wb')
        pickle.dump(config.betas_adj_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        config.betas_adj_missed_dict = config.betas_missed_dict
        f = open(fn_missed_dict, 'wb')
        pickle.dump(config.betas_missed_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        exog_dict = {}

        if 'cells' in data_params:

            cells_dict = load_cells_dict(config)

            if isinstance(data_params['cells'], list):
                all_types = list(cells_dict.keys())
                for key in all_types:
                    if key not in data_params['cells']:
                        cells_dict.pop(key)

                if len(list(cells_dict.keys())) != len(data_params['cells']):
                    raise ValueError(f'Wrong number of cells types.')

                exog_dict.update(cells_dict)

        if 'observables' in data_params:

            observables_dict = load_observables_dict(config)
            if isinstance(data_params['observables'], list):
                all_types = list(observables_dict.keys())
                for key in all_types:
                    if key not in data_params['observables']:
                        observables_dict.pop(key)

                if len(list(observables_dict.keys())) != len(data_params['observables']):
                    raise ValueError(f'Wrong number of observables types.')

                exog_dict.update(observables_dict)

        num_cpgs = config.betas_data.shape[0]
        num_subjects = config.betas_data.shape[1]
        config.betas_adj_data = np.zeros((num_cpgs, num_subjects), dtype=np.float32)

        for cpg, row in tqdm(config.betas_dict.items(), mininterval=60.0, desc='betas_adj_data creating'):

            raw_betas = config.betas_data[row, :]
            current_exog_dict = copy.deepcopy(exog_dict)

            if len(config.betas_missed_dict[cpg]) > 0:

                for key in current_exog_dict:
                    values = []
                    for value_id in range(0, len(current_exog_dict[key])):
                        if value_id not in config.betas_adj_missed_dict[cpg]:
                            values.append(current_exog_dict[key][value_id])
                    current_exog_dict[key] = values

                betas = []
                passed_ids = []
                for beta_id in range(0, len(raw_betas)):
                    if beta_id not in config.betas_adj_missed_dict[cpg]:
                        betas.append(raw_betas[beta_id])
                        passed_ids.append(beta_id)
            else:

                betas = raw_betas
                passed_ids = list(range(0, len(betas)))

            endog_dict = {cpg: betas}
            endog_df = pd.DataFrame(endog_dict)
            exog_df = pd.DataFrame(current_exog_dict)

            reg_res = sm.OLS(endog=endog_df, exog=exog_df).fit()

            residuals = list(map(np.float32, reg_res.resid))
            betas_adj = residuals + np.mean(betas)
            betas_adj_raw = np.zeros(num_subjects, dtype=np.float32)
            for beta_id in range(0, len(passed_ids)):
                betas_adj_raw[passed_ids[beta_id]] = betas_adj[beta_id]

            for missed_id in config.betas_adj_missed_dict[cpg]:
                betas_adj_raw[missed_id] = np.float32('nan')

            config.betas_adj_data[row] = betas_adj

        np.savez_compressed(fn_data, data=config.betas_adj_data)

        # Clear data
        del config.betas_data
