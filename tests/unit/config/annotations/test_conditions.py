import unittest
from tests.definitions import ROOT_DIR
from pydnameth import Data
from pydnameth import Experiment
from pydnameth import Annotations
from pydnameth import Observables
from pydnameth import Cells
from pydnameth import Attributes
from pydnameth import Config
from pydnameth.infrastucture.load.excluded import load_excluded
from pydnameth.infrastucture.load.annotations import load_annotations_dict
from pydnameth.config.annotations.conditions import \
    exclude_condition,\
    check_condition,\
    global_check,\
    cpg_name_condition
from tests.tear_down import clear_cache


class TestAnnotationsConditions(unittest.TestCase):

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
                'Class': ['ClassA', 'ClassB']
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

    def tearDown(self):
        clear_cache(self.config)

    def test_exclude_condition(self):
        self.config.excluded = load_excluded(self.config)
        self.config.annotations_dict = {self.config.annotations.id_name: [['cg00000165']]}
        condition = exclude_condition(self.config, 0)
        self.assertEqual(False, condition)

    def test_snp_condition(self):
        self.config.annotations_dict = {'Probe_SNPs': [['rs12616624']],
                                        'Probe_SNPs_10': [['rs35342923']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(False, condition)

    def test_gene_region_condition(self):
        self.config.annotations_dict = {'UCSC_REFGENE_NAME': [['PRR4', 'TAS2R20']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(True, condition)

        self.config.annotations_dict = {'UCSC_REFGENE_NAME': [[]]}
        condition = check_condition(self.config, 0)
        self.assertEqual(False, condition)

    def test_chr_condition(self):
        self.config.annotations_dict = {'CHR': [['X']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(False, condition)

        self.config.annotations_dict = {'CHR': [['1']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(True, condition)

    def test_probe_class_condition(self):
        self.config.annotations_dict = {'Class': [['ClassA']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(True, condition)

        self.config.annotations_dict = {'Class': [['ClassB']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(True, condition)

        self.config.annotations_dict = {'Class': [['ClassC']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(False, condition)

        self.config.annotations_dict = {'Class': [['ClassD']]}
        condition = check_condition(self.config, 0)
        self.assertEqual(False, condition)

    def test_check_conditions(self):
        self.config.excluded = load_excluded(self.config)

        self.config.annotations_dict = {
            self.config.annotations.id_name: [['cg00001249']],
            'CROSS_R': [['0']],
            'Probe_SNPs': [[]],
            'Probe_SNPs_10': [[]],
            'CHR': [['14']],
            'UCSC_REFGENE_NAME': [['PRR4', 'TAS2R20']],
            'RELATION_TO_UCSC_CPG_ISLAND': [['S_Shelf']],
            'Class': [['ClassA']]
        }
        condition = global_check(self.config, 0)
        self.assertEqual(True, condition)

        self.config.annotations_dict['CHR'] = [['X']]
        condition = global_check(self.config, 0)
        self.assertEqual(False, condition)

    def get_count(self, condition):
        count = 0
        for item_id in range(0, len(self.config.annotations_dict[self.config.annotations.id_name])):
            if condition(self.config, item_id):
                count += 1
        return count

    def test_count_exclude_condition(self):
        self.config.excluded = load_excluded(self.config)
        self.config.annotations_dict = load_annotations_dict(self.config)
        count = self.get_count(exclude_condition)
        self.assertEqual(count, 297)

    def test_count_snp_condition(self):
        self.config.annotations_dict = load_annotations_dict(self.config)
        target_keys = [self.config.annotations.id_name, 'Probe_SNPs', 'Probe_SNPs_10']
        new_dict = {}
        for key in target_keys:
            new_dict[key] = self.config.annotations_dict[key]
        self.config.annotations_dict = new_dict
        count = self.get_count(check_condition)
        self.assertEqual(count, 250)

    def test_count_gene_region_condition(self):
        self.config.annotations_dict = load_annotations_dict(self.config)
        target_keys = [self.config.annotations.id_name, 'UCSC_REFGENE_NAME']
        new_dict = {}
        for key in target_keys:
            new_dict[key] = self.config.annotations_dict[key]
        self.config.annotations_dict = new_dict
        count = self.get_count(check_condition)
        self.assertEqual(count, 292)

    def test_count_probe_class_condition(self):
        self.config.annotations_dict = load_annotations_dict(self.config)
        target_keys = [self.config.annotations.id_name, 'Class']
        new_dict = {}
        for key in target_keys:
            new_dict[key] = self.config.annotations_dict[key]
        self.config.annotations_dict = new_dict
        count = self.get_count(check_condition)
        self.assertEqual(count, 202)

    def test_count_check_conditions(self):
        self.config.excluded = load_excluded(self.config)
        self.config.annotations_dict = load_annotations_dict(self.config)
        count = self.get_count(global_check)
        self.assertEqual(count, 151)

    def test_cpg_name(self):
        self.config.annotations_dict = {self.config.annotations.id_name: [['cg00001269']]}
        condition = cpg_name_condition(self.config, 0)
        self.assertEqual(True, condition)

        self.config.annotations_dict = {self.config.annotations.id_name: [[]]}
        condition = cpg_name_condition(self.config, 0)
        self.assertEqual(False, condition)

    def test_cross_reactive_condition(self):
        self.config.annotations_dict = load_annotations_dict(self.config)
        target_keys = [self.config.annotations.id_name, 'CROSS_R']
        new_dict = {}
        for key in target_keys:
            new_dict[key] = self.config.annotations_dict[key]
        self.config.annotations_dict = new_dict
        count = self.get_count(check_condition)
        self.assertEqual(count, 278)

    def test_get_num_NS_chr(self):
        self.config.annotations_dict = load_annotations_dict(self.config)
        target_keys = [self.config.annotations.id_name, 'CHR']
        new_dict = {}
        for key in target_keys:
            new_dict[key] = self.config.annotations_dict[key]
        self.config.annotations_dict = new_dict
        count = self.get_count(check_condition)
        self.assertEqual(count, 276)


if __name__ == '__main__':
    unittest.main()
