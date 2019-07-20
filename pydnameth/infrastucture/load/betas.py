from pydnameth.infrastucture.path import get_data_base_path
import numpy as np
import os.path
import pickle
from tqdm import tqdm


def get_line_list(line):
    line_list = line.split('\t')
    for val_id in range(0, len(line_list)):
        line_list[val_id] = line_list[val_id].replace('"', '').rstrip()
    return line_list


def load_betas(config):

    suffix = ''
    if bool(config.experiment.data_params):
        suffix += '_' + str(config.experiment.get_data_params_str())

    fn_dict = get_data_base_path(config) + '/' + 'betas_dict' + suffix + '.pkl'
    fn_missed_dict = get_data_base_path(config) + '/' + 'betas_missed_dict' + suffix + '.pkl'
    fn_data = get_data_base_path(config) + '/' + 'betas' + suffix
    fn_txt = fn_data + '.txt'
    fn_npz = fn_data + '.npz'

    if os.path.isfile(fn_dict) and os.path.isfile(fn_dict) and os.path.isfile(fn_npz):

        f = open(fn_dict, 'rb')
        config.betas_dict = pickle.load(f)
        f.close()

        f = open(fn_missed_dict, 'rb')
        config.betas_missed_dict = pickle.load(f)
        f.close()

        data = np.load(fn_npz)
        config.betas_data = data['data']

    else:

        config.betas_dict = {}
        config.betas_missed_dict = {}
        config.betas_missed_dict['any'] = []

        f = open(fn_txt)
        f.readline()
        cpg_id = 0
        for line in tqdm(f, mininterval=60.0, desc='betas_dict creating'):
            line_list = get_line_list(line)
            cpg = line_list[0]
            betas = line_list[1::]

            missed_indexes = []
            for missed_value in config.annotations.missed_values:
                indexes = [i for i, x in enumerate(betas) if x == missed_value]
                missed_indexes += indexes
            missed_indexes.sort()
            config.betas_missed_dict[cpg] = missed_indexes

            config.betas_dict[cpg] = cpg_id
            cpg_id += 1
        f.close()

        f = open(fn_dict, 'wb')
        pickle.dump(config.betas_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        f = open(fn_missed_dict, 'wb')
        pickle.dump(config.betas_missed_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        num_cpgs = cpg_id

        f = open(fn_txt)
        header_line = f.readline()
        headers = get_line_list(header_line)
        subjects = headers[1:len(headers)]

        config.betas_data = np.zeros((num_cpgs, len(subjects)), dtype=np.float32)

        cpg_id = 0
        for line in tqdm(f, mininterval=60.0, desc='betas_data creating'):
            line_list = get_line_list(line)
            cpg = line_list[0]
            betas = line_list[1::]
            for beta_id in range(0, len(betas)):
                if beta_id in config.betas_missed_dict[cpg]:
                    betas[beta_id] = np.float32('nan')
                else:
                    betas[beta_id] = np.float32(betas[beta_id])
            config.betas_data[cpg_id] = betas
            cpg_id += 1
        f.close()

        np.savez_compressed(fn_npz, data=config.betas_data)
