from pydnameth.config.experiment.types import Task, Method, DataType
from pydnameth.scripts.develop.plot import plot_scatter, plot_scatter_comparison


def epimutations_plot_scatter(
    data,
    annotations,
    attributes,
    observables_list,
    method_params=None
):
    plot_scatter(
        DataType.epimutations,
        data,
        annotations,
        attributes,
        observables_list,
        child_method=Method.linreg,
        method_params=method_params
    )


def epimutations_plot_scatter_comparison(
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
        data_type=DataType.epimutations,
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
