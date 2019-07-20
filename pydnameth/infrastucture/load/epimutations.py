from pydnameth.infrastucture.load.betas import load_betas
from pydnameth.infrastucture.path import get_data_base_path
import numpy as np
import os.path
from tqdm import tqdm
import math


def load_epimutations(config):

    suffix = ''
    if bool(config.experiment.data_params):
        suffix += '_' + str(config.experiment.get_data_params_str())

    fn_data = get_data_base_path(config) + '/' + 'epimutations' + suffix

    config.epimutations_list = ['epimutations']
    config.epimutations_dict = {'epimutations': 0}
    config.epimutations_missed_dict = {'any': []}

    config.experiment.data_params = {}
    load_betas(config)

    if os.path.isfile(fn_data + '.npz'):

        data = np.load(fn_data + '.npz')
        config.epimutations_data = data['data']

    else:

        num_cpgs = config.betas_data.shape[0]
        num_subjects = config.betas_data.shape[1]
        config.epimutations_data = np.zeros((num_cpgs, num_subjects), dtype=np.int)

        for cpg, row in tqdm(config.betas_dict.items(), mininterval=60.0):

            betas_raw = config.betas_data[row, :]
            if len(config.betas_missed_dict[cpg]) > 0:
                betas = []
                for beta_id in range(0, len(betas_raw)):
                    if beta_id not in config.betas_missed_dict[cpg]:
                        betas.append(betas_raw[beta_id])
            else:
                betas = betas_raw

            quartiles = np.percentile(betas, [25, 75])
            iqr = quartiles[1] - quartiles[0]
            left = quartiles[0] - (3.0 * iqr)
            right = quartiles[1] + (3.0 * iqr)

            curr_row = np.zeros(num_subjects, dtype=np.int)
            for subject_id in range(0, num_subjects):
                curr_point = betas_raw[subject_id]
                if not math.isnan(curr_point):
                    if curr_point < left or curr_point > right:
                        curr_row[subject_id] = 1

            config.epimutations_data[row] = curr_row

        np.savez_compressed(fn_data + '.npz', data=config.epimutations_data)
        np.savetxt(fn_data + '.txt', config.epimutations_data, delimiter='\t', fmt='%d')

    # Clear data
    del config.betas_data
