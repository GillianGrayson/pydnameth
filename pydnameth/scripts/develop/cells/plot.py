from pydnameth.config.experiment.types import Method, DataType
from pydnameth.scripts.develop.plot import plot_scatter


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
