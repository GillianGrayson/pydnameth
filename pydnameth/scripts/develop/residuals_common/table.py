from pydnameth.config.experiment.types import DataType, Method
from pydnameth.scripts.develop.table import table_aggregator_linreg, table_aggregator_variance, table


def residuals_table_aggregator_linreg(
    data,
    annotations,
    attributes,
    observables_list,
    data_params,
):
    table_aggregator_linreg(
        DataType.residuals,
        data,
        annotations,
        attributes,
        observables_list,
        data_params=data_params,
    )


def residuals_table_aggregator_variance(
    data,
    annotations,
    attributes,
    observables_list,
    data_params,
):
    table_aggregator_variance(
        DataType.residuals,
        data,
        annotations,
        attributes,
        observables_list,
        data_params=data_params,
    )


def residuals_table_oma(
    data,
    annotations,
    attributes,
    data_params,
):
    table(
        data=data,
        annotations=annotations,
        attributes=attributes,
        data_type=DataType.residuals,
        method=Method.oma,
        data_params=data_params,
        task_params=None,
        method_params=None
    )
