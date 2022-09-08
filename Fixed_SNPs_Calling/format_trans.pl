#!usr/bin/perl
use warnings;

while(<>){
chomp;
if($_=~m/^Chrom/){
}else{
@a=split "\t",$_;
print "$a[0]\t$a[1]\t$a[2]\t$a[3]\t";
@b=split ":",$a[4];
@c=split ":",$a[5];
print "$b[4]\t$b[1]\t";
if($#c==6){
printf "%-15s\t","$b[2]=$c[2]:$c[3]";
printf "%-15s\t","$b[3]=$c[4]:$c[5]";
printf "%-20s\t","$c[0]:$c[1]=$c[6]";
print "$a[6]\t$a[7]\t$a[8]\t$a[9]\n";
}
elsif($#c==5){
printf "%-15s\t","$b[2]=$c[1]:$c[2]";
printf "%-15s\t","$b[3]=$c[3]:$c[4]";
printf "%-20s\t","$c[0]=$c[5]";
print "$a[6]\t$a[7]\t$a[8]\t$a[9]\n";
#print "$b[4]\t$b[1]\t$b[2]($c[1]:$c[2])\t$b[3]($c[3]:$c[4])\t$c[0]=$c[5]\t$a[6]\t$a[7]\t$a[8]\t$a[9]\n"
}
}
}
