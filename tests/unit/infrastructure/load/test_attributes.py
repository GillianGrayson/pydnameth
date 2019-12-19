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
from pydnameth.infrastucture.load.attributes import load_observables_dict
from pydnameth.infrastucture.path import get_data_base_path
from tests.tear_down import clear_cache


class TestLoadAnnotations(unittest.TestCase):

    def setUp(self):
        data = Data(
            path=ROOT_DIR,
            base='fixtures'
        )

        experiment = Experiment(
            data=None,
            task=None,
            method=None
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

    def tearDown(self):
        clear_cache(self.config)

    def test_load_observables_dict_num_elems(self):
        observables_dict = load_observables_dict(self.config)
        self.assertEqual(len(observables_dict['age']), 729)

    def test_load_observables_dict_age_range(self):
        observables_dict = load_observables_dict(self.config)
        self.assertEqual(max(observables_dict['age']) - min(observables_dict['age']), 80)

    def test_load_observables_dict_check_pkl_file_creation(self):
        load_observables_dict(self.config)

        create = os.path.isfile(get_data_base_path(self.config) + '/' + self.config.attributes.observables.name + '.pkl')

        self.assertEqual(True, create)

    def test_load_observables_dict_check_sum_smoke(self):
        observables_dict = load_observables_dict(self.config)

        sum_smoke = sum(list(map(int, observables_dict['smoke'])))

        self.assertEqual(188, sum_smoke)

    def test_load_observables_dict_num_Male(self):
        observables_dict = load_observables_dict(self.config)

        indexes = [ind for ind, val in enumerate(observables_dict['gender']) if val == 'M']

        self.assertEqual(341, len(indexes))

    def test_load_observables_dict_num_Female(self):
        observables_dict = load_observables_dict(self.config)

        indexes = [ind for ind, val in enumerate(observables_dict['gender']) if val == 'F']

        self.assertEqual(388, len(indexes))


if __name__ == '__main__':
    unittest.main()
