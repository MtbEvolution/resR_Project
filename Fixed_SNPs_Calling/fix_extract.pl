#!usr/bin/perl
use warnings;

while(<>){
chomp;
@a=split "\t",$_;
$a[4]=~s/%//;
if($a[4]>=90 && $a[5]>=10){
print "$_\n";
}
}
