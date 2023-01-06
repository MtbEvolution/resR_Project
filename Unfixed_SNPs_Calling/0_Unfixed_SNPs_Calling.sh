#mapping fastq files to genome template; starting from paired-end fastq files ($i, $j); for single-end fastq files, change the parameters accordingly.
sickle pe -l 35 -f $i -r $j -t sanger -o $fq1 -p $fq2 -s $fq3; 
bwa aln -R 1 ~/template/bwa/tb.ancestor.fasta $fq1 > $sai1; 
bwa aln -R 1 ~/template/bwa/tb.ancestor.fasta $fq2 > $sai2;
bwa aln -R 1 ~/template/bwa/tb.ancestor.fasta $fq3 > $sai3;
bwa sampe -a 1000 -n 1 -N 1 ~/template/bwa/tb.ancestor.fasta $sai1 $sai2 $fq1 $fq2 > $samp;
bwa samse -n 1 ~/template/bwa/tb.ancestor.fasta $sai3 $fq3 > $sams;
samtools view -bhSt ~/template/bwa/tb.ancestor.fasta.fai $samp -o $bamp;
samtools view -bhSt ~/template/bwa/tb.ancestor.fasta.fai $sams -o $bams; 
samtools merge $bamm $bamp $bams;
samtools sort $bamm -o $sortbam; 
samtools index $sortbam;
#generate mpileup file for each sample
samtools mpileup -q 30 -Q 20 -ABOf ~/template/bwa/tb.ancestor.fasta $sortbam > $pileup; 
#using varscan for SNP calling
java -jar ~/bin/VarScan.jar mpileup2snp $pileup --min-coverage 3 --min-reads2 2 --min-avg-qual 20 --min-var-freq 0.01 --min-freq-for-hom 0.9 --p-value 99e-02 --strand-filter 1 > $var; 
java -jar ~/bin/VarScan.jar mpileup2cns $pileup --min-coverage 3 --min-avg-qual 20 --min-var-freq 0.75 --min-reads2 2 --strand-filter 1 > $cns;
#exclude regions belonging to PPE/PE and insertion sequences, and also exclude the regions that were recently marked as error-prone (Marin, Bioinformatics, 2022)
perl ~/script/PE_IS_filt.pl ~/script/Excluded_loci_mask.list $var > $ppe;
perl ~/script/format_trans.pl $ppe > $format;
#extract read location from mpileup file (where does a mutation allele locate on a seqeuncing read), for further filtering based on tail distribution
perl ~/script/mix_pileup_merge.pl $format $pileup > $forup; 
#average sequencing depth, only include samples with genome coverage rate >0.9 and sequencing depth >20X
sed 's/:/\t/g' $cns|awk '{if (\$6 >= 3){n++;sum+=\$6}} END {print \"\t\",n/4411532,\"\t\",sum/n}' > $dep
#extract unfixed SNPs from forup files, this will create two files: "markdisc" and "markkept"; the suspected false positives(such as mutations with tail region enrichment) will be moved to markdisc file
perl ~/script/mix_extract_0.95.pl $forup > $mix;
perl ~/script/forup_format.pl $mix > $mixfor;
perl ~/script/info_mark.pl $mixfor > $mixmark;
perl ~/script/redepin_filt.pl Excluded_loci_mask.list $dep $mixmark
#filter list of highly repeated mutations with similar mutational frequency
#for those unfixed mutations that arise >=5 times in the 50K isolates, further check their reliability based on 1) the ratio in "markkept"; 2) the distribution of the mutational frequency.
cat *mixmarkkept > all_KEPT.txt; perl ~/script/loci_freq_count.pl all_KEPT.txt >kept_repeat.txt
cat *mixmark > all_MIX.txt;perl ~/script/loci_freq_count.pl all_MIX.txt > mix_repeat.txt
perl ~/script/repeat_number_merge.pl mix_repeat.txt kept_repeat.txt > merge_kept_mix.txt
perl ~/script/ratio.pl merge_kept_mix.txt > merge_kept_mix_ratio.txt
awk '$4>=5' merge_kept_mix_ratio.txt |awk '$6>0.6'|cut -f1|while read i;do echo $i > $i.per5up.txt;grep -w $i all_KEPT.txt|cut -f12 >> $i.per5up.txt;done
paste *per5up.txt > 5up_0.6_paste.txt
perl ~/script/stdv.pl 5up_0.6_paste.txt |awk '$2<0.25'|cut -f1 > 5up_0.6_0.25.list
perl ~/script/freq_extract.pl 5up_0.6_0.25.list 5up_0.6_paste.txt > 5up_0.6_0.25.txt
awk '$4>=5' merge_kept_mix_ratio.txt|cut -f1 > 5up.list
perl ~/script/repeat_loci.pl 5up_0.6_0.25.list 5up.list > 5up_remove_loc.list 
perl ~/script/unfix_scripts/repeatloci_filter.pl 5up_remove_loc.list $markkept > $keptfilt
#annotation of unfixed SNPs
cut -f9-11 $keptfilt > $keptsnp
perl ~/script/1_MTBC_Annotation_mtbc_4411532.pl $keptsnp > $keptanofilt
