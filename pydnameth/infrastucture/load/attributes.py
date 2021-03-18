from pydnameth.routines.common import is_float
from pydnameth.infrastucture.path import get_data_base_path
from pydnameth.routines.common import categorize_data
from pydnameth.routines.common import is_categorical
import numpy as np
import os.path
import pickle
import pandas as pd


def load_observables_dict(config):
    fn = get_data_base_path(config) + '/' + config.attributes.observables.name
    fn_txt = fn + '.txt'
    fn_xlsx = fn + '.xlsx'
    fn_pkl = fn + '.pkl'

    if os.path.isfile(fn_pkl):

        f = open(fn_pkl, 'rb')
        observables_dict = pickle.load(f)
        f.close()

    else:

        if os.path.isfile(fn_xlsx):
            df = pd.read_excel(fn_xlsx, engine='openpyxl')
            tmp_dict = df.to_dict()
            observables_dict = {}
            for key in tmp_dict:
                curr_dict = tmp_dict[key]
                observables_dict[key] = list(curr_dict.values())

        elif os.path.isfile(fn_txt):
            f = open(fn_txt)
            key_line = f.readline()
            keys = key_line.split('\t')
            keys = [x.rstrip() for x in keys]

            observables_dict = {}
            for key in keys:
                observables_dict[key] = []

            for line in f:
                values = line.split('\t')
                for key_id in range(0, len(keys)):
                    key = keys[key_id]
                    value = values[key_id].rstrip()
                    if is_float(value):
                        value = float(value)
                        if value.is_integer():
                            observables_dict[key].append(int(value))
                        else:
                            observables_dict[key].append(float(value))
                    else:
                        observables_dict[key].append(value)
            f.close()

        else:
            raise ValueError('No observables file')

        f = open(fn_pkl, 'wb')
        pickle.dump(observables_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    return observables_dict


def load_observables_categorical_dict(config):
    fn = get_data_base_path(config) + '/' + config.attributes.observables.name + '_categorical'
    fn_pkl = fn + '.pkl'

    if os.path.isfile(fn_pkl):

        f = open(fn_pkl, 'rb')
        observables_categorical_dict = pickle.load(f)
        f.close()

    else:

        observables_categorical_dict = {}

        if config.observables_dict is not None:
            observables_dict = config.observables_dict
        else:
            observables_dict = load_observables_dict(config)

        na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>',
                     'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', '-', '--']

        for key in observables_dict:
            all_numeric = True
            for i in range(0, len(observables_dict[key])):
                value = observables_dict[key][i]
                if value in na_values:
                    value = np.nan
                if is_float(value):
                    value = float(value)
                    if value.is_integer():
                        observables_dict[key][i] = value
                    else:
                        observables_dict[key][i] = float(value)
                else:
                    observables_dict[key][i] = value
                    all_numeric = False
            if all_numeric:
                observables_categorical_dict[key] = np.asarray(config.observables_dict[key])
            else:
                observables_categorical_dict[key] = categorize_data(np.asarray(config.observables_dict[key]))

        f = open(fn_pkl, 'wb')
        pickle.dump(observables_categorical_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    return observables_categorical_dict


def get_is_observables_categorical(config):

    is_observables_categorical = {}

    if config.observables_dict is not None:
        observables_dict = config.observables_dict
    else:
        observables_dict = load_observables_dict(config)

    for key, values in observables_dict.items():
        if is_categorical(values):
            is_observables_categorical[key] = True
        else:
            is_observables_categorical[key] = False

    return is_observables_categorical


def load_cells_dict(config):
    fn = get_data_base_path(config) + '/' + config.attributes.cells.name
    fn_txt = fn + '.txt'
    fn_pkl = fn + '.pkl'
    fn_xlsx = fn + '.xlsx'



    if os.path.isfile(fn_pkl):

        f = open(fn_pkl, 'rb')
        cells_dict = pickle.load(f)
        f.close()

    else:

        if os.path.isfile(fn_xlsx):
            df = pd.read_excel(fn_xlsx, engine='openpyxl')
            tmp_dict = df.to_dict()
            cells_dict = {}
            for key in tmp_dict:
                curr_dict = tmp_dict[key]
                cells_dict[key] = list(curr_dict.values())

        elif os.path.isfile(fn_txt):

            f = open(fn_txt)
            key_line = f.readline()
            keys = key_line.split('\t')
            keys = [x.rstrip() for x in keys]

            cells_dict = {}
            for key in keys:
                cells_dict[key] = []

            for line in f:
                values = line.split('\t')
                for key_id in range(0, len(keys)):
                    key = keys[key_id]
                    value = values[key_id].rstrip()
                    if is_float(value):
                        cells_dict[key].append(float(value))
                    else:
                        cells_dict[key].append(value)
            f.close()

            f = open(fn_pkl, 'wb')
            pickle.dump(cells_dict, f, pickle.HIGHEST_PROTOCOL)
            f.close()

        else:
            return None

    return cells_dict
