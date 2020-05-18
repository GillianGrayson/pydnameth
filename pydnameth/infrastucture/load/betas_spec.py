from pydnameth.infrastucture.load.betas import load_betas, get_line_list
from pydnameth.infrastucture.path import get_data_base_path
import numpy as np
import os.path
from tqdm import tqdm
import pandas as pd


def load_betas_spec(config):

    file = config.experiment.data_params.pop('file', None)
    filename, file_extension = os.path.splitext(file)
    fn_list = get_data_base_path(config) + '/' + file
    fn_data_csv = get_data_base_path(config) + '/' + filename + '.csv'

    if os.path.isfile(fn_list):
        f = open(fn_list)
        cpgs = f.readlines()
        cpgs = [x.rstrip() for x in cpgs]
        f.close()
    else:
        raise ValueError('File with cpgs is not exists.')

    if not os.path.isfile(fn_data_csv):

        indexes = config.attributes_indexes

        fn_betas = get_data_base_path(config) + '/' + 'betas.txt'
        f = open(fn_betas)
        header_line = f.readline()
        headers = get_line_list(header_line)
        subject_headers = list(np.array(headers[1::])[indexes])
        f.close()

        load_betas(config)

        num_cols = len(subject_headers) + 1  # header col
        num_rows = len(cpgs) + 1  # header row

        betas = np.zeros((num_rows, num_cols), dtype=object)
        row_id = 0
        betas[row_id] = ['ProbeID'] + subject_headers
        row_id += 1

        for cpg in tqdm(cpgs, mininterval=60.0, desc='betas_spec creating'):
            if cpg in config.betas_dict:
                cpg_row_id = config.betas_dict[cpg]
                curr_betas = list(np.array(config.betas_data[cpg_row_id])[indexes])
                for missed_id in config.betas_missed_dict[cpg]:
                    curr_betas[missed_id] = 'NaN'
                line = [cpg] + curr_betas
                betas[row_id] = line
            else:
                line = [cpg] + ['NaN'] * (num_cols - 1)
                betas[row_id] = line

            row_id += 1

        pd.DataFrame(betas).to_csv(fn_data_csv, index=False, header=False)
