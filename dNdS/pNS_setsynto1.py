#!/usr/bin/env python

#Calculate pNS for SNPs detected in each gene.

#Modified from https://github.com/swisstph/TBRU_serialTB 2_pNS.py

################
# DEPENDENCIES #
################

import os
import pickle
import pandas as pd
import numpy as np
import scipy.stats as ss
import random
from collections import Counter
import sys
import re

#############################
#  LOAD PRE-PROCESSED DATA  #
#############################

CWD = os.getcwd() #get current working directory

PROCESSED_PATH = os.path.abspath(os.path.join(CWD))
#INTERIM_PATH = os.path.abspath(os.path.join(CWD, os.pardir, os.pardir, 'data', 'interim'))

ALL = pd.read_csv(sys.argv[1],index_col=0)
#PATIENT_DATA = pickle.load(open('{}{}ALLELE_info.pkl'.format(PROCESSED_PATH, os.sep),'rb'))
#ANNOTATION_DICT = pickle.load(open('{}{}2_unfixed_annotation.pkl'.format(INTERIM_PATH, os.sep),'rb'))

#CCDC5079_genome = 4414325 #based on NC_021251 (CCDC5079)

#Using the approach of the four-fold degenerate primers, I estimated that
#the probability of specific mutations is (the details of the calculations
#are shown in /notebooks/1.5_ATR_pNS_calculations.ipynb

# A -> T 0.0238095238095
# A -> C 0.297619047619
# A -> G 0.678571428571
# C -> A 0.116666666667
# C -> T 0.644444444444
# C -> G 0.238888888889
# G -> A 0.619565217391
# G -> T 0.141304347826
# G -> C 0.239130434783
# T -> A 0.0285714285714
# T -> C 0.7
# T -> G 0.271428571429

#Transform into a dictionary

MUT_PROB = {'A': {'C': {'FFD_FIXED': 0.297619047619},
                  'G': {'FFD_FIXED': 0.678571428571},
                  'T': {'FFD_FIXED': 0.0238095238095}},
            'C': {'A': {'FFD_FIXED': 0.116666666667},
                  'G': {'FFD_FIXED': 0.238888888889},
                  'T': {'FFD_FIXED': 0.644444444444}},
            'G': {'A': {'FFD_FIXED': 0.619565217391},
                  'C': {'FFD_FIXED': 0.239130434783},
                  'T': {'FFD_FIXED': 0.141304347826}},
            'T': {'A': {'FFD_FIXED': 0.0285714285714},
                  'C': {'FFD_FIXED': 0.7},
                  'G': {'FFD_FIXED': 0.271428571429}}
           }

#Define the codon table
CODON_TABLE = {'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAT': 'N', 'ACA': 'T',
               'ACC': 'T', 'ACG': 'T', 'ACT': 'T', 'AGA': 'R', 'AGC': 'S',
               'AGG': 'R', 'AGT': 'S', 'ATA': 'I', 'ATC': 'I', 'ATG': 'M',
               'ATT': 'I', 'CAA': 'Q', 'CAC': 'H', 'CAG': 'Q', 'CAT': 'H',
               'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P', 'CGA': 'R',
               'CGC': 'R', 'CGG': 'R', 'CGT': 'R', 'CTA': 'L', 'CTC': 'L',
               'CTG': 'L', 'CTT': 'L', 'GAA': 'E', 'GAC': 'D', 'GAG': 'E',
               'GAT': 'D', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
               'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G', 'GTA': 'V',
               'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'TAA': 'STOP',
               'TAC': 'Y', 'TAG': 'STOP', 'TAT': 'Y','TCA': 'S',
               'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TGA': 'STOP',
               'TGC': 'C', 'TGG': 'W', 'TGT': 'C', 'TTA': 'L', 'TTC': 'F',
               'TTG': 'L', 'TTT': 'F'}

#Derive a function for mutating codons
def mutate_codon(codon, mutation_pool):
    """Randomly mutate a base within a codon using
    a predefined substitution matrix. Score the
    translational impact of mutaitons.

    INPUT:
    ------
    codon: str, NNN where N in ['A', 'C', 'G', 'T']
    mutation_pool: dict, key: base, value: str

    OUTPUT:
    ------
    int, 0|1, 0 for synonymous mutation, 1 for nonsynonymous

    NOTES:
    ------
    uses the standard random library
    """

    pos = random.randint(0,2)
    random_base = random.choice(mutation_pool[codon[pos]])

    cl = list(codon)
    cl[pos] = random_base
    new_codon = ''.join(cl)

    CODON_TABLE = {'AAA': 'K', 'AAC': 'N', 'AAG': 'K', 'AAT': 'N', 'ACA': 'T',
                   'ACC': 'T', 'ACG': 'T', 'ACT': 'T', 'AGA': 'R', 'AGC': 'S',
                   'AGG': 'R', 'AGT': 'S', 'ATA': 'I', 'ATC': 'I', 'ATG': 'M',
                   'ATT': 'I', 'CAA': 'Q', 'CAC': 'H', 'CAG': 'Q', 'CAT': 'H',
                   'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P', 'CGA': 'R',
                   'CGC': 'R', 'CGG': 'R', 'CGT': 'R', 'CTA': 'L', 'CTC': 'L',
                   'CTG': 'L', 'CTT': 'L', 'GAA': 'E', 'GAC': 'D', 'GAG': 'E',
                   'GAT': 'D', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
                   'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G', 'GTA': 'V',
                   'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'TAA': 'STOP',
                   'TAC': 'Y', 'TAG': 'STOP', 'TAT': 'Y','TCA': 'S',
                   'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TGA': 'STOP',
                   'TGC': 'C', 'TGG': 'W', 'TGT': 'C', 'TTA': 'L', 'TTC': 'F',
                   'TTG': 'L', 'TTT': 'F'}

    return int(CODON_TABLE[codon]!=CODON_TABLE[new_codon])

#Derive the expected rate of non-synonymous substitutions by simulations
#based on parameters extracted from looking at four-fold degenerate codons.

MUTATION_POOL = {'A': '',
                 'C': '',
                 'G': '',
                 'T': ''
                }

for k in MUTATION_POOL.keys():
    for k1,v1 in MUT_PROB[k].items():
        MUTATION_POOL[k]+=(k1*int(v1['FFD_FIXED']*1000))

NSY_EXPECTATION = {
    codon: np.mean([np.mean([mutate_codon(codon, MUTATION_POOL)\
                             for x in range(1000)]) for y in range(100)])
    for codon in CODON_TABLE.keys()
}

#Save the expectation
pickle.dump(NSY_EXPECTATION,
            open('{}{}NSY_EXPECTATION.pkl'.format(PROCESSED_PATH,os.sep),'wb'))


#Calculate pNS for each time-point
#First mark mutations occurring in CODING regions of the genome
ALL['CODING'] = [int(x in ['SYN', 'NSY']) for x in ALL.SNP_TYPE]

pNS_VAR = {}

click = 0
for gene in np.unique(ALL.GENE):
    #count observed data
    oN = Counter(ALL.SNP_TYPE[(ALL.GENE==gene)&(ALL.CODING==1)])['NSY']
    oS = Counter(ALL.SNP_TYPE[(ALL.GENE==gene)&(ALL.CODING==1)])['SYN']
    Total_mut = oN + oS
    if oS == 0: #set syn mutation number to 1 when no syn mutation detected
        oS = 1
    #use mutation matrix to derive the expectation
    #print(gene)
    eN = np.sum([NSY_EXPECTATION[x] for x in ALL.WT_CODON[
                (ALL.GENE==gene)&
                (ALL.CODING==1)]])
    eS = (oN+oS)-eN
    #calculate pNS
    try:
        pNS = (oN/eN)/(oS/eS)

    except:
        pNS = np.nan
    
    #Remove all calculations where less than 3 mutations 
    #detected
    #if Total_mut <= 3:
    #    pNS = np.nan

    #Remove all calculations where there were no observed mutations
    #in either category
    if pNS==0:
        pNS = np.nan
    if pNS==np.inf:
        pNS = np.nan

    #use binomial random sampling to derive a null distribution of pNS
    bN = np.sum([ss.binom(p=NSY_EXPECTATION[x],n=1).rvs(random_state=seed) for (seed,x) in enumerate(
                ALL.WT_CODON[(ALL.GENE==gene)&(ALL.CODING==1)])])
    bS = (oN+oS)-bN
    try:
        pNS_b = (bN/eN)/(bS/eS)

    except:
        pNS_b = np.nan

    #Remove all calculations where there were no observed mutations
    #in either category
    if pNS_b==0:
        pNS_b = np.nan
    if pNS_b==np.inf:
        pNS_b = np.nan



    pNS_VAR[click] = {'GENE': gene,
                      'pNS': pNS,
                      'OBSERVED_SYN': oS,
                      'OBSERVED_NSY': oN,
                      'EXPECTED_SYN': eS,
                      'EXPECTED_NSY': eN,
                      'NEUTRAL_SYN': bS,
                      'NEUTRAL_NSY': bN,
                      'pNS_NEUTRAL': pNS_b,
                      'TOTAL': oS+oN
                     }
    click+=1

#Remove the temporary CODING category.
ALL.drop('CODING', axis=1, inplace=True)

#Reformat the resulting dictionary into a pandas DataFrame
pNS_df = pd.DataFrame(pNS_VAR).T

csvout = sys.argv[1]
csvout = re.sub('ann.csv', 'gene.csv', csvout)
#Save the output to interim outputs.
pNS_df.to_csv(csvout)
