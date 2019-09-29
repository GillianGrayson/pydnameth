from pydnameth.infrastucture.path import get_save_path
from pydnameth.infrastucture.file_name import get_file_name
import pandas as pd
import os.path
import pickle


def load_table_dict(config):
    table_dict = load_table_dict_pkl(config)
    return table_dict


def load_table_dict_xlsx(config):
    fn = get_save_path(config) + '/' + get_file_name(config) + '.xlsx'
    if os.path.isfile(fn):
        df = pd.read_excel(fn)
        tmp_dict = df.to_dict()
        table_dict = {}
        for key in tmp_dict:
            curr_dict = tmp_dict[key]
            table_dict[key] = list(curr_dict.values())
        return table_dict
    else:
        raise IOError(f'No such file: {fn}')


def load_table_dict_pkl(config):
    fn = get_save_path(config) + '/' + get_file_name(config) + '.pkl'
    if os.path.isfile(fn):
        f = open(fn, 'rb')
        table_dict = pickle.load(f)
        f.close()
        return table_dict
    else:
        raise IOError(f'No such file: {fn}')
