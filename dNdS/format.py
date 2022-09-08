import sys
import re
from collections import defaultdict
from itertools import product

#homo = defaultdict(list)

#with open(sys.argv[1]) as f: #2005_homo_site.txt
#    for line in f:
#        cols = line.strip().split()
#        (pos, alt) = cols[2].strip().split('_')
#        homo[pos].append([alt,cols[3]])

print 'LOCUS,REF,ALT,SNP_TYPE,GENE,TRANSITION,WT_CODON,IND_TIME'

pl = ['A', 'G']
md = ['C', 'T']

transversion = {}
for (x,y) in list(product(pl, md)):
    key = x+y
    transversion[key] = 1
    keyr = y+x
    transversion[keyr] = 1

with open(sys.argv[1]) as af: #SNP annotation file
    for line in af:
        cols = line.strip().split()
        mut = cols[1].strip()+cols[2].strip()
        trans = '0' if mut in transversion else '1' #Transition (1)/Transversion (0)
        gene = cols[6]
        
        if re.match('Non', cols[4]):
            snp_type = 'NSY'
            wt_codon = cols[5].strip().split('-')[0]
        elif re.match('Syn', cols[4]):
            snp_type = 'SYN'
            wt_codon = cols[5].strip().split('-')[0]
        elif re.match('-', cols[4]):
            snp_type = 'IGR'
            wt_codon = '-'
        ref = cols[1]
        alt = cols[2]
        int_time = '1'
        #if cols[0] in homo:
        #    for x in homo[cols[0]]:
        #        if alt == x[0]:
        #            int_time = str(x[1])
        print cols[0]+','+ref+','+alt+','+snp_type+','\
        +gene+','+trans+','+wt_codon+','+int_time
