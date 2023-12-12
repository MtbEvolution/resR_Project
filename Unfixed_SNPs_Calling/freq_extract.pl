#!usr/bin/perl
use warnings;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
#@a=split "\t",$_;
$hash{$_}=1;
}
close F1;

my $n=0;
open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
@a=split "\t",$_;
if($n==0){
foreach $i(0..$#a){
if(exists $hash{$a[$i]}){
print "$a[$i]\t";
$loci{$i}=1;
}
}
$n++;
}else{
foreach $i(0..$#a){
if(exists $loci{$i}){
print "$a[$i]\t";
}
}
}
print "\n";
}
close F2;

