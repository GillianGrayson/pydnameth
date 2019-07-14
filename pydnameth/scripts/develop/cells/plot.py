from pydnameth.config.experiment.types import Method, DataType
from pydnameth.scripts.develop.plot import plot_scatter, plot_scatter_comparison


def cells_plot_scatter(
    data,
    annotations,
    attributes,
    observables_list,
    child_method=Method.linreg,
    method_params=None
):

    plot_scatter(
        data_type=DataType.cells,
        data=data,
        annotations=annotations,
        attributes=attributes,
        observables_list=observables_list,
        child_method=child_method,
        method_params=method_params
    )


def cells_plot_scatter_comparison(
    data_list,
    annotations_list,
    attributes_list,
    observables_list,
    data_params_list,
    rows_dict,
    cols_dict,
    child_method=Method.linreg,
    method_params=None,
):
    plot_scatter_comparison(
        data_type=DataType.cells,
        data_list=data_list,
        annotations_list=annotations_list,
        attributes_list=attributes_list,
        observables_list=observables_list,
        data_params_list=data_params_list,
        rows_dict=rows_dict,
        cols_dict=cols_dict,
        child_method=child_method,
        method_params=method_params,
    )
