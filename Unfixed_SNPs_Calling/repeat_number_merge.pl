#!usr/bin/perl
use warnings;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
$id="$a[0]\t$a[1]\t$a[2]";
$hash{$id}=$a[3];
}
close F1;

open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
@a=split "\t",$_;
$mut="$a[0]\t$a[1]\t$a[2]";
print "$_\t$hash{$mut}\n";
}
close F2;

