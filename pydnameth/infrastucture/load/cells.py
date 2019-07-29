def load_cells(config):
    config.cells_list = list(config.cells_dict.keys())
    config.cells_missed_dict = {}
    for cells in config.cells_list:
        config.cells_missed_dict[cells] = []
