from enum import Enum


class AnnotationKey(Enum):
    chr = 'CHR'
    map_info = 'MAPINFO'
    Probe_SNPs = 'Probe_SNPs'
    Probe_SNPs_10 = 'Probe_SNPs_10'
    gene = 'UCSC_REFGENE_NAME'
    probe_class = 'Class'
    geo = 'RELATION_TO_UCSC_CPG_ISLAND'
    bop = 'BOP'
    cross_reactive = 'CROSS_R'
