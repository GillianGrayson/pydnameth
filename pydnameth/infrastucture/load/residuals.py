from pydnameth.infrastucture.load.betas import load_betas
from pydnameth.infrastucture.path import get_data_base_path
from pydnameth.infrastucture.load.attributes import load_cells_dict, load_observables_dict
from pydnameth.routines.common import is_categorical
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pickle
import os.path
from tqdm import tqdm
import copy


def load_residuals_common(config):

    suffix = ''
    if bool(config.experiment.data_params):
        data_params = config.experiment.data_params
        suffix += '_' + config.experiment.get_data_params_str()
    else:
        raise ValueError(f'Exog for residuals is empty.')

    fn_dict = get_data_base_path(config) + '/' + 'residuals_dict' + suffix + '.pkl'
    fn_missed_dict = get_data_base_path(config) + '/' + 'residuals_missed_dict' + suffix + '.pkl'
    fn_data = get_data_base_path(config) + '/' + 'residuals' + suffix + '.npz'

    if os.path.isfile(fn_dict) and os.path.isfile(fn_data):

        f = open(fn_dict, 'rb')
        config.residuals_dict = pickle.load(f)
        f.close()

        f = open(fn_missed_dict, 'rb')
        config.residuals_missed_dict = pickle.load(f)
        f.close()

        data = np.load(fn_data)
        config.residuals_data = data['data']

    else:

        config.experiment.data_params = {}
        load_betas(config)

        config.residuals_dict = config.betas_dict
        f = open(fn_dict, 'wb')
        pickle.dump(config.residuals_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        config.residuals_missed_dict = config.betas_missed_dict
        f = open(fn_missed_dict, 'wb')
        pickle.dump(config.residuals_missed_dict, f, pickle.HIGHEST_PROTOCOL)
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

        exog_keys = []
        for exog_type, exog_data in exog_dict.items():
            if is_categorical(exog_data):
                exog_keys.append('C(' + exog_type + ')')
            else:
                exog_keys.append(exog_type)
        formula = 'cpg ~ ' + ' + '.join(exog_keys)

        exog_keys = []
        for exog_type, exog_data in exog_dict.items():
            if is_categorical(exog_data):
                exog_keys.append('C(' + exog_type + ')')
            else:
                exog_keys.append(exog_type)
        formula = 'cpg ~ ' + ' + '.join(exog_keys)

        num_cpgs = config.betas_data.shape[0]
        num_subjects = config.betas_data.shape[1]
        config.residuals_data = np.zeros((num_cpgs, num_subjects), dtype=np.float32)

        for cpg, row in tqdm(config.betas_dict.items(), mininterval=60.0, desc='residuals_data creating'):
            raw_betas = config.betas_data[row, :]

            current_data_dict = copy.deepcopy(exog_dict)

            if len(config.betas_missed_dict[cpg]) > 0:

                for key in current_data_dict:
                    values = []
                    for value_id in range(0, len(current_data_dict[key])):
                        if value_id not in config.betas_adj_missed_dict[cpg]:
                            values.append(current_data_dict[key][value_id])
                    current_data_dict[key] = values

                betas = []
                passed_ids = []
                for beta_id in range(0, len(raw_betas)):
                    if beta_id not in config.betas_adj_missed_dict[cpg]:
                        betas.append(raw_betas[beta_id])
                        passed_ids.append(beta_id)
            else:

                betas = raw_betas
                passed_ids = list(range(0, len(betas)))

            current_data_dict['cpg'] = betas
            data_df = pd.DataFrame(current_data_dict)
            reg_res = smf.ols(formula=formula, data=data_df).fit()

            residuals = list(map(np.float32, reg_res.resid))
            residuals_raw = np.zeros(num_subjects, dtype=np.float32)
            for beta_id in range(0, len(passed_ids)):
                residuals_raw[passed_ids[beta_id]] = residuals[beta_id]

            for missed_id in config.residuals_missed_dict[cpg]:
                residuals_raw[missed_id] = np.float32('nan')

            config.residuals_data[row] = residuals

        np.savez_compressed(fn_data, data=config.residuals_data)

        # Clear data
        del config.betas_data
