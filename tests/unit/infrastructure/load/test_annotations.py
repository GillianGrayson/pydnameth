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
from pydnameth.infrastucture.load.annotations import load_annotations_dict
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
            method=None,
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

    def test_load_annotations_dict_num_elems(self):
        annotations_dict = load_annotations_dict(self.config)
        self.assertEqual(300, len(annotations_dict['ID_REF']), )

    def test_load_annotations_dict_num_keys(self):
        annotations_dict = load_annotations_dict(self.config)
        self.assertEqual(14, len(list(annotations_dict.keys())))

    def test_load_annotations_dict_num_chrs(self):
        annotations_dict = load_annotations_dict(self.config)
        all_elems = []
        for elems in annotations_dict['CHR']:
            for elem in elems:
                all_elems.append(elem)
        self.assertEqual(11, len(set(all_elems)))

    def test_load_annotations_dict_num_bops(self):
        annotations_dict = load_annotations_dict(self.config)
        all_elems = []
        for elems in annotations_dict['BOP']:
            for elem in elems:
                all_elems.append(elem)
        self.assertEqual(81, len(set(all_elems)))

    def test_load_annotations_check_pkl_file_creation(self):
        load_annotations_dict(self.config)
        create = os.path.isfile(get_data_base_path(self.config) + '/' + self.config.annotations.name + '.pkl')
        self.assertEqual(True, create)

    def test_load_annotations_num_cross_r_cpgs(self):
        annotations_dict = load_annotations_dict(self.config)
        all_elems = []
        for elems in annotations_dict['CROSS_R']:
            all_elems.append(elems[0])
        num_of_cross_r_cpg = sum(list(map(int, all_elems)))
        self.assertEqual(22, num_of_cross_r_cpg)

    def test_load_annotations_dict_num_geo(self):
        annotations_dict = load_annotations_dict(self.config)
        all_elems = []
        for elems in annotations_dict['RELATION_TO_UCSC_CPG_ISLAND']:
            for elem in elems:
                all_elems.append(elem)
        self.assertEqual(5, len(set(all_elems)))

    def test_load_annotations_dict_num_class(self):
        annotations_dict = load_annotations_dict(self.config)
        all_elems = []
        for elems in annotations_dict['Class']:
            for elem in elems:
                all_elems.append(elem)
        self.assertEqual(4, len(set(all_elems)))


if __name__ == '__main__':
    unittest.main()
