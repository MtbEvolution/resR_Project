#!usr/bin/perl
use warnings;

open F1,$ARGV[0] or die $!;#open PPE_INS_loci.list
while(<F1>){
chomp;
$hash{$_}=1;
}
close F1;

open F2,$ARGV[1] or die $!;#open snp list
while(<F2>){
chomp;
@b=split "\t",$_;
if(!exists $hash{$b[1]}){
print "$_\n";
}
}
close F2;
