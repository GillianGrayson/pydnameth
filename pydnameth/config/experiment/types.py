from enum import Enum


class Task(Enum):
    table = 'table'
    clock = 'clock'
    plot = 'plot'
    create = 'create'

    def __str__(self):
        return str(self.value)


class Method(Enum):
    linreg = 'linreg'
    heteroscedasticity = 'heteroscedasticity'
    variance_linreg = 'variance_linreg'
    variance = 'variance'
    cluster = 'cluster'
    histogram = 'histogram'
    scatter = 'scatter'
    scatter_comparison = 'scatter_comparison'
    curve = 'curve'
    polygon = 'polygon'
    special = 'special'
    z_test_linreg = 'z_test_linreg'
    variance_histogram = 'variance_histogram'
    aggregator = 'aggregator'
    mock = 'mock'
    regular = 'regular'
    range = 'range'

    def __str__(self):
        return str(self.value)


class DataType(Enum):
    betas = 'betas'
    betas_adj = 'betas_adj'
    betas_horvath_calculator = 'betas_horvath_calculator'
    betas_spec = 'betas_spec'
    residuals_common = 'residuals_common'
    residuals_special = 'residuals_special'
    epimutations = 'epimutations'
    entropy = 'entropy'
    observables = 'observables'
    suppl = 'suppl'
    cache = 'cache'

    def __str__(self):
        return str(self.value)
