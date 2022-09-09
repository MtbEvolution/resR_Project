#!usr/bin/perl
use warnings;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
foreach $i($a[3]..$a[4]){
$hash{$i}=1;
}
}
close F1;

open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
if($_=~m/^@/){
print "$_\n";
}else{
@a=split "\t",$_;
if(!exists $hash{$a[3]}){
print "$_\n";
}
}
}
close F2;

