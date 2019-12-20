# -*- coding: utf-8 -*-

"""Top-level package for pydnameth."""
# flake8: noqa

__author__ = """Aaron Blare"""
__email__ = 'aaron.blare@mail.ru'
__version__ = '0.2.4'

from .config.config import Config
from .config.common import CommonTypes
from .config.annotations.annotations import Annotations
from .config.annotations.types import AnnotationKey
from .config.attributes.attributes import Cells, Observables, Attributes
from .config.data.data import Data
from .config.data.types import DataPath, DataBase
from .config.experiment.experiment import Experiment
from .config.experiment.types import DataType, Task, Method

from pydnameth.scripts.develop.betas.table import \
    betas_table_aggregator_linreg,\
    betas_table_aggregator_variance,\
    betas_table_linreg, \
    betas_table_heteroscedasticity, \
    betas_table_oma, \
    betas_table_cluster, \
    betas_table_variance
from pydnameth.scripts.develop.betas.clock import \
    betas_clock_linreg,\
    betas_clock_special
from pydnameth.scripts.develop.betas.plot import \
    betas_plot_scatter,\
    betas_plot_curve_clock,\
    betas_plot_variance_histogram, \
    betas_plot_scatter_comparison

from pydnameth.scripts.develop.betas_horvath_calculator.create import \
    betas_horvath_calculator_create_regular

from pydnameth.scripts.develop.betas_spec.create import \
    betas_spec_create_regular

from pydnameth.scripts.develop.epimutations.load import \
    epimutations_load
from pydnameth.scripts.develop.epimutations.table import \
    epimutations_table_z_test_linreg, \
    epimutations_table_ancova, \
    epimutations_table_aggregator_linreg, \
    epimutations_table_aggregator_variance
from pydnameth.scripts.develop.epimutations.plot import \
    epimutations_plot_scatter,\
    epimutations_plot_scatter_comparison

from pydnameth.scripts.develop.entropy.plot import \
    entropy_plot_scatter, \
    entropy_plot_scatter_comparison
from pydnameth.scripts.develop.entropy.table import \
    entropy_table_z_test_linreg, \
    entropy_table_ancova, \
    entropy_table_aggregator_linreg, \
    entropy_table_aggregator_variance

from pydnameth.scripts.develop.observables.plot import \
    observables_plot_histogram

from pydnameth.scripts.develop.cells.plot import \
    cells_plot_scatter, \
    cells_plot_scatter_comparison
from pydnameth.scripts.develop.cells.table import \
    cells_table_z_test_linreg, \
    cells_table_ancova, \
    cells_table_aggregator_linreg, \
    cells_table_aggregator_variance

from pydnameth.scripts.develop.residuals.plot import \
    residuals_plot_scatter, \
    residuals_plot_scatter_comparison
from pydnameth.scripts.develop.residuals.table import \
    residuals_table_aggregator_linreg,\
    residuals_table_aggregator_variance, \
    residuals_table_oma, \
    residuals_table_approach_3

from pydnameth.scripts.develop.betas_adj.plot import \
    betas_adj_plot_scatter, \
    betas_adj_plot_scatter_comparison
from pydnameth.scripts.develop.betas_adj.table import \
    betas_adj_table_aggregator_linreg,\
    betas_adj_table_aggregator_variance, \
    betas_adj_table_oma, \
    betas_adj_table_approach_3

from pydnameth.scripts.develop.genes.plot import \
    genes_plot_scatter, \
    genes_plot_scatter_comparison
from pydnameth.scripts.develop.genes.table import \
    genes_table_aggregator_linreg,\
    genes_table_aggregator_variance
