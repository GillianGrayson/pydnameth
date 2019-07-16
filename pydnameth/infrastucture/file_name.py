def get_file_name(config):
    file_name = ''
    file_name += config.experiment.get_method_params_str()
    return file_name
