from pydnameth.config.experiment.types import DataType


def get_file_name(config):
    file_name = ''
    if config.experiment.data == DataType.cells:
        file_name += 'cells(' + str(config.attributes.cells) + ')_'
    file_name += config.experiment.get_method_params_str()
    return file_name
