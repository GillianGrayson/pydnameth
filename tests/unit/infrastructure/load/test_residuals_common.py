import unittest
import os
from tests.definitions import ROOT_DIR
from pydnameth.config.data.data import Data
from pydnameth.config.experiment.experiment import Experiment
from pydnameth.config.annotations.annotations import Annotations
from pydnameth.config.attributes.attributes import Observables
from pydnameth.config.attributes.attributes import Cells
from pydnameth.config.attributes.attributes import Attributes
from pydnameth.config.config import Config
from pydnameth.infrastucture.load.residuals import load_residuals
from tests.tear_down import clear_cache
from pydnameth.infrastucture.path import get_data_base_path


class TestLoadResidualsCommon(unittest.TestCase):

    def setUp(self):

        data = Data(
            path=ROOT_DIR,
            base='fixtures'
        )

        data_params = {'cells': ['B', 'CD4T', 'NK', 'CD8T', 'Gran']}

        experiment = Experiment(
            data=None,
            task=None,
            method=None,
            data_params=data_params
        )

        annotations = Annotations(
            name='annotations',
            type='450k',
            exclude='excluded',
            select_dict={
                'CROSS_R': ['0'],
                'Probe_SNPs': ['empty'],
                'Probe_SNPs_10': ['empty'],
                'CHR': ['-X', '-Y'],
                'UCSC_REFGENE_NAME': ['non-empty'],
            }
        )

        observables = Observables(
            name='observables',
            types={}
        )

        cells = Cells(
            name='cells',
            types='any'
        )

        attributes = Attributes(
            target='age',
            observables=observables,
            cells=cells
        )

        self.config = Config(
            data=data,
            experiment=experiment,
            annotations=annotations,
            attributes=attributes,
            is_run=True,
            is_root=True
        )
        self.config.initialize()

    def test_load_residuals_check_files_creation(self):
        suffix = '_' + self.config.experiment.get_data_params_str()
        fn_dict = get_data_base_path(self.config) + '/' + 'residuals_dict' + suffix + '.pkl'
        fn_data = get_data_base_path(self.config) + '/' + 'residuals' + suffix + '.npz'

        load_residuals(self.config)

        self.assertEqual(True, os.path.isfile(fn_dict) and os.path.isfile(fn_data))

    def tearDown(self):
        clear_cache(self.config)

    def test_load_residuals_check_len_cpg_dict(self):
        load_residuals(self.config)
        self.assertEqual(300, len(list(self.config.residuals_dict)))

    def test_load_residuals_check_shape_cpg_data(self):
        load_residuals(self.config)
        self.assertEqual((300, 729), self.config.residuals_data.shape)
