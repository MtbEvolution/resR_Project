tophat2 -p 1 --library-type fr-unstranded -G ~/template/bowtie2/H37Rv.gtf -o rnaseqdata.tp.out ~/template/bowtie2/H37Rv rnaseqdata.fastq.gz; 
samtools view -h -o rnaseqdata.tp.out/accepted_hits.sam rnaseqdata.tp.out/accepted_hits.bam; 
perl ~/script/RNA_seq/remove_rRNA_reads.pl ~/template/bowtie2/H37Rv.rRNA.gff rnaseqdata.tp.out/accepted_hits.sam > rnaseqdata.tp.out/accepted_hits_filt.sam; 
samtools view -Sb rnaseqdata.tp.out/accepted_hits_filt.sam > rnaseqdata.tp.out/accepted_hits_filt.bam; 
cufflinks -p 4 -G ~/template/bowtie2/H37Rv.gtf --library-type fr-unstranded -o rnaseqdata.clf.out rnaseqdata.tp.out/accepted_hits_filt.bam
