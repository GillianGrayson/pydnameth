from pydnameth.infrastucture.path import get_cache_path
from pydnameth.config.annotations.conditions import global_check
import copy
import os.path
import pickle
import numpy as np


def subset_annotations(config):
    aux_data_fn = get_cache_path(config) + '/' + 'aux_data.pkl'

    if config.annotations.type == '450k':

        if os.path.isfile(aux_data_fn):
            f = open(aux_data_fn, 'rb')
            aux_data = pickle.load(f)
            f.close()
            config.cpg_list = aux_data['cpg_list']
            config.cpg_gene_dict = aux_data['cpg_gene_dict']
            config.cpg_bop_dict = aux_data['cpg_bop_dict']
            config.gene_cpg_dict = aux_data['gene_cpg_dict']
            config.gene_bop_dict = aux_data['gene_bop_dict']
            config.bop_cpg_dict = aux_data['bop_cpg_dict']
            config.bop_gene_dict = aux_data['bop_gene_dict']
            config.cpg_map_info_dict = aux_data['cpg_map_info_dict']
        else:
            config.cpg_list = []
            config.cpg_gene_dict = {}
            config.cpg_bop_dict = {}
            config.gene_cpg_dict = {}
            config.gene_bop_dict = {}
            config.bop_cpg_dict = {}
            config.bop_gene_dict = {}
            config.cpg_map_info_dict = {}

            cpgs_all = config.annotations_dict[config.annotations.id_name]
            genes_all = config.annotations_dict['UCSC_REFGENE_NAME']
            bops_all = config.annotations_dict['BOP']
            map_infos_all = config.annotations_dict['MAPINFO']

            for index, cpg in enumerate(cpgs_all):

                if global_check(config, index):

                    cpg = cpgs_all[index][0]
                    config.cpg_list.append(cpg)

                    map_info = map_infos_all[index][0]
                    if map_info == 'NA':
                        map_info = '0'
                    config.cpg_map_info_dict[cpg] = int(map_info)

                    genes = genes_all[index]
                    if len(genes) > 0:
                        config.cpg_gene_dict[cpg] = genes
                        for gene in genes:
                            if gene in config.gene_cpg_dict:
                                config.gene_cpg_dict[gene].append(cpg)
                            else:
                                config.gene_cpg_dict[gene] = [cpg]

                    bops = bops_all[index]
                    if len(bops) > 0:
                        config.cpg_bop_dict[cpg] = bops
                        for bop in bops:
                            if bop in config.bop_cpg_dict:
                                config.bop_cpg_dict[bop].append(cpg)
                            else:
                                config.bop_cpg_dict[bop] = [cpg]

                    if len(genes) > 0 and len(bops) > 0:
                        for gene in genes:
                            if gene in config.gene_bop_dict:
                                config.gene_bop_dict[gene] += bops
                            else:
                                config.gene_bop_dict[gene] = copy.deepcopy(bops)
                        for bop in bops:
                            if bop in config.bop_gene_dict:
                                config.bop_gene_dict[bop] += genes
                            else:
                                config.bop_gene_dict[bop] = copy.deepcopy(genes)

            # Sorting cpgs by map_info in gene dict
            for gene, cpgs in config.gene_cpg_dict.items():
                map_infos = []
                for cpg in cpgs:
                    map_infos.append(int(config.cpg_map_info_dict[cpg]))
                order = np.argsort(map_infos)
                cpgs_sorted = list(np.array(cpgs)[order])
                config.gene_cpg_dict[gene] = cpgs_sorted

            # Sorting cpgs by map_info in bop dict
            for bop, cpgs in config.bop_cpg_dict.items():
                map_infos = []
                for cpg in cpgs:
                    map_infos.append(int(config.cpg_map_info_dict[cpg]))
                order = np.argsort(map_infos)
                cpgs_sorted = list(np.array(cpgs)[order])
                config.bop_cpg_dict[bop] = cpgs_sorted

            aux_data = {
                'cpg_list': config.cpg_list,
                'cpg_gene_dict': config.cpg_gene_dict,
                'cpg_bop_dict': config.cpg_bop_dict,
                'gene_cpg_dict': config.gene_cpg_dict,
                'gene_bop_dict': config.gene_bop_dict,
                'bop_cpg_dict': config.bop_cpg_dict,
                'bop_gene_dict': config.bop_gene_dict,
                'cpg_map_info_dict': config.cpg_map_info_dict
            }

            f = open(aux_data_fn, 'wb')
            pickle.dump(aux_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    elif config.annotations.type == '850k':

        if os.path.isfile(aux_data_fn):
            f = open(aux_data_fn, 'rb')
            aux_data = pickle.load(f)
            f.close()
            config.cpg_list = aux_data['cpg_list']
            config.cpg_gene_dict = aux_data['cpg_gene_dict']
            config.gene_cpg_dict = aux_data['gene_cpg_dict']
            config.cpg_map_info_dict = aux_data['cpg_map_info_dict']
            config.bops = aux_data['bops']

        else:
            config.cpg_list = []
            config.cpg_gene_dict = {}
            config.gene_cpg_dict = {}
            config.cpg_map_info_dict = {}

            cpgs_all = config.annotations_dict[config.annotations.id_name]
            genes_all = config.annotations_dict['UCSC_RefGene_Name']
            map_infos_all = config.annotations_dict['MAPINFO']
            chr_all = config.annotations_dict['CHR']
            geo_all = config.annotations_dict['UCSC_CpG_Islands_Name']
            geo_type_all = config.annotations_dict['Relation_to_UCSC_CpG_Island']

            config.bops = {}

            for index, cpg in enumerate(cpgs_all):

                if global_check(config, index):

                    cpg = cpgs_all[index][0]
                    config.cpg_list.append(cpg)

                    map_info = map_infos_all[index]
                    if len(map_info) > 0:
                        map_info = map_info[0]
                        if map_info == 'NA':
                            map_info = '0'
                    else:
                        map_info = 0
                    config.cpg_map_info_dict[cpg] = int(map_info)

                    genes = genes_all[index]
                    genes.sort()
                    if len(genes) > 0:
                        config.cpg_gene_dict[cpg] = genes
                        for gene in genes:
                            if gene in config.gene_cpg_dict:
                                config.gene_cpg_dict[gene].append(cpg)
                            else:
                                config.gene_cpg_dict[gene] = [cpg]

                    if len(chr_all[index]) > 0:
                        chr = chr_all[index][0]
                    else:
                        chr = ''
                    if len(geo_all[index]) > 0:
                        geo = geo_all[index][0]
                    else:
                        geo = ''
                    if len(geo_type_all[index]) > 0:
                        geo_type = geo_type_all[index][0]
                    else:
                        geo_type = ''

                    if geo == '':
                        if len(genes) > 0:
                            bop_class = 'C'
                            bop_names = [f'{chr}*{gene}' for gene in genes]
                        else:
                            bop_class = 'D'
                            bop_names = [cpg]
                    else:
                        if len(genes) > 0:
                            bop_class = 'A'
                            bop_names = [f'{geo}*{geo_type}']
                        else:
                            bop_class = 'B'
                            bop_names = [f'{geo}*{geo_type}*nogene']

                    for bop_name in bop_names:
                        if bop_name in config.bops:
                            if config.bops[bop_name]['class'] != bop_class:
                                raise ValueError(f'Error: Different classes in BOP creation: {bop_name}')
                            config.bops[bop_name]['cpg'].append(cpg)
                            config.bops[bop_name]['map_info'].append(map_info)
                            config.bops[bop_name]['gene'].update(set(genes))
                        else:
                            config.bops[bop_name] = {}
                            config.bops[bop_name]['class'] = bop_class
                            config.bops[bop_name]['cpg'] = [cpg]
                            config.bops[bop_name]['map_info'] = [map_info]
                            config.bops[bop_name]['gene'] = set(genes)

            # Sorting cpgs by map_info in gene dict
            for gene, cpgs in config.gene_cpg_dict.items():
                map_infos = []
                for cpg in cpgs:
                    map_infos.append(int(config.cpg_map_info_dict[cpg]))
                order = np.argsort(map_infos)
                cpgs_sorted = list(np.array(cpgs)[order])
                config.gene_cpg_dict[gene] = cpgs_sorted

            # Sorting cpgs by map_info in bop
            for bop_name, bop_dict in config.bops.items():
                cpg = config.bops[bop_name]['cpg']
                map_info = config.bops[bop_name]['map_info']
                order = np.argsort(map_info)
                cpg_sorted = list(np.array(cpg)[order])
                config.bops[bop_name]['cpg'] = cpg_sorted

            aux_data = {
                'cpg_list': config.cpg_list,
                'cpg_gene_dict': config.cpg_gene_dict,
                'gene_cpg_dict': config.gene_cpg_dict,
                'cpg_map_info_dict': config.cpg_map_info_dict,
                'bops': config.bops
            }

            f = open(aux_data_fn, 'wb')
            pickle.dump(aux_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    elif config.annotations.type == 'epityper':

        if os.path.isfile(aux_data_fn):
            f = open(aux_data_fn, 'rb')
            aux_data = pickle.load(f)
            f.close()
            config.cpg_list = aux_data['cpg_list']
        else:
            config.cpg_list = []
            cpgs_all = config.annotations_dict[config.annotations.id_name]
            for index, cpg in enumerate(cpgs_all):
                cpg = cpgs_all[index][0]
                config.cpg_list.append(cpg)

            aux_data = {
                'cpg_list': config.cpg_list,
            }

            f = open(aux_data_fn, 'wb')
            pickle.dump(aux_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
