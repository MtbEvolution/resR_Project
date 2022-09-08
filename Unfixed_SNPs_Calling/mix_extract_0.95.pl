#!usr/bin/perl
use warnings;

# 1) extract unfix mutations from forup file;
# 2) frequency filter, real reads filter, minimum reads filter, forward and reverse stran filter;

while(<>){
@a=split "\t",$_;
$a[4]=~s/%//;
if($a[4]<=95){   # mutation frequency <= 95%
@b=split "=",$a[7];
@c=split "=",$a[6];
$real=$b[0]+$c[0]; 
if($real/$a[5] > 0.8){ # real reads ratio > 80%
if($b[0]>=4 && $c[0]>=4){  # minimum reads number >= 4 for both wild-type and mutant alleles
@d=split ":",$b[1];
@e=split ":",$c[1];
if(($d[0]>=1)&&($d[1]>=1) && ($e[0]>=1)&&($e[1]>=1)){  #forward and reverse strand should both >= 1 for both wild-type and mutant alleles
print "$_";
}
}
}
}
}
