------------------------
"pNS"
------------------------
#/n/holyscratch01/sfortune_lab/edwin/50K_Data/a_analysis/3_pns


--------------------------------------------------
报错总结：2725571,G,A,SYN,Rv2427A,1,C,1
  File "/home/edwin/script/pNS/pNS_setsynto1.py", line 171, in <module>
    (ALL.CODING==1)]])
  File "/home/edwin/script/pNS/pNS_setsynto1.py", line 169, in <listcomp>
    eN = np.sum([NSY_EXPECTATION[x] for x in ALL.WT_CODON[
KeyError: 'C'
原因是因为有个位点(oxyR)被注释成codon但是没有提供codon的密码子，只给了个C
--------------------------------------------------


$1 把所有样本ano文件合并在一起
cd L212;cd dr;ls|grep ano|while read a;do cat $a;done > L212_r_ano.txt;sort L212_r_ano.txt|uniq > L212_r_ano_uniq.txt;
cd ..;cd ds;ls|grep ano|while read a;do cat $a;done > L212_s_ano.txt;sort L212_s_ano.txt|uniq > L212_s_ano_uniq.txt

$2 格式转换#Transfor the uniq ano file to ann.csv format
python2.7 ~/script/pNS/format.py L1.uniq.ano1 > L1.uniq.ann.csv 

$3 计算pNpS，这一步需在服务器上完成 #Split ann.csv to each individual gene, and calculate pNS separately.
perl ~/script/pns/gene_split.pl all.uniq.ann.csv #生成每个基因的uniq.ann.csv文件
rm *-*csv MTB*
ls |grep uniq.ann.csv|while read i;do echo "python3 ~/script/pNS/pNS_setsynto1.py $i";done > run.sh  #每个基因单独计算PNS，删除基因间"rm *-*"
    perl ~/script/multi_process.pl run.sh 100
    cp ~/script/sbatch.template.sh .
    vi sbatch.template.sh
    
    ls Process_*sh|while read i;do cat sbatch.template.sh $i > PNSSBATCH_${i%.sh}.sh;done
    ls PNSSBATCH*.sh|while read i;do sbatch $i;done

$4 #extract pNS value and cat into one file
ls|grep 1_|while read i;do cd $i;ls |grep gene.csv|while read j;do perl ~/script/pns/pns_extract.pl $j;done > ${i}_pns.txt;cd ..;done



"1940以前突变的pNS分析"
#/Users/edwin/Work/PROJECTS/1_New_DR_Gene/Evolution_Trajectory/51_Long-term_Selection/5_1940_before_snp_age/
40-SNP阈值
    perl ../ano_filt_4pns.pl ../filt.list ../all_ann_cat.txt ../All_nodelength.txt 40 > ano_40.txt
    python2.7 ~/script/pNS/format.py ano_40.txt > 40.uniq.ann.csv 
    python3 ~/script/pNS/pNS_setsynto1.py 40.uniq.ann.csv #this will create "40.uniq.gene.csv" file
    perl ~/script/pns/result_format.pl ~/script/translate/mtbc_translate/2_Tuberculist_new_20150307 40.uniq.gene.csv > 40.uniq.pnps.txt   
50-SNP阈值 #on clusters
    perl ano_filt_4pns.pl filt.list all_ann_cat.txt All_nodelength.txt 50 > ano_50.txt
    python2.7 ~/script/pNS/format.py ano_50.txt > 50.uniq.ann.csv 
    python3 ~/script/pNS/pNS_setsynto1.py 50.uniq.ann.csv 
    perl ~/script/pns/result_format.pl ~/script/translate/mtbc_translate/2_Tuberculist_new_20150307 50.uniq.gene.csv > 50.uniq.pnps.txt
60-SNP阈值 #on clusters
    perl ano_filt_4pns.pl filt.list all_ann_cat.txt All_nodelength.txt 60 > ano_60.txt
    python2.7 ~/script/pNS/format.py ano_60.txt > 60.uniq.ann.csv 
    python3 ~/script/pNS/pNS_setsynto1.py 60.uniq.ann.csv 
    perl ~/script/pns/result_format.pl ~/script/translate/mtbc_translate/2_Tuberculist_new_20150307 60.uniq.gene.csv > 60.uniq.pnps.txt
    
"1940以后突变的pNS分析"
#/Users/edwin/Work/PROJECTS/1_New_DR_Gene/Evolution_Trajectory/51_Long-term_Selection/52_1940_after_snp
    python2.7 ~/script/pNS/format.py all_ann_cat_filt.txt > after.uniq.ann.csv 
    python3 ~/script/pNS/pNS_setsynto1.py after.uniq.ann.csv #on cluster
    perl ~/script/pns/result_format.pl ~/script/translate/mtbc_translate/2_Tuberculist_new_20150307 after.uniq.gene.csv > after.uniq.pnps.txt


"合并dNdS与pNpS结果"
perl merge.pl before.dnds.txt before.pnps.txt after.dnds.txt after.pnps.txt > 1_merge_bef_aft.txt
cat 67_corrected.list|while read i;do grep -w $i 1_merge_bef_aft.txt ;done > 67_corrected_dnds_pnps.txt
cat 68_corrected.list|while read i;do grep -w $i 1_merge_bef_aft.txt ;done > 68_corrected_dnds_pnps.txt
perl format_forR.pl 68_corrected_dnds_pnps.txt > 68_corrected_dnds_pnps_for_R.txt
perl filter.pl 02_exclude.list 68_corrected_dnds_pnps_for_R.txt > 68_corrected_dnds_pnps_for_R_include.txt
perl order.pl 02_exclude.list 00_order.list  #生成R作图的坐标顺序






