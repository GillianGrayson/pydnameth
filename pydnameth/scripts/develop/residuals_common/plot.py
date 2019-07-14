from pydnameth.config.experiment.types import Method, DataType
from pydnameth.scripts.develop.plot import plot_scatter, plot_scatter_comparison


def residuals_common_plot_scatter(
    data,
    annotations,
    attributes,
    observables_list,
    child_method=Method.linreg,
    data_params=None,
    method_params=None
):
    plot_scatter(
        DataType.residuals_common,
        data,
        annotations,
        attributes,
        observables_list,
        child_method,
        data_params,
        method_params
    )


def residuals_common_plot_scatter_comparison(
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
        data_type=DataType.residuals_common,
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
