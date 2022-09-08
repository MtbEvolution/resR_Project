library("seqinr")
options(max.print = 99999999)

s <- read.alignment(file="gyrA.fa", format="fasta")
sink("gyrA.kaks.txt")
kaks(s)
sink()


temp=list.files(pattern = ".fa")
for (i in 1:length(temp)){
  s<-read.alignment(temp[i],format="fasta")
    name <- temp[i]
    sink(paste(name,".txt", sep=""))
    a <- kaks(s)
    print(a)
    sink()
}