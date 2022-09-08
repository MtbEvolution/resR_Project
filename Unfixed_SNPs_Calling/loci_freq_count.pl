#!usr/bin/perl
use warnings;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
$id="$a[8]\t$a[9]\t$a[10]";
if(!exists $hash{$id}){
$hash{$id}=1;
}else{
$hash{$id}++;
}
}
close F1;

foreach $i(keys %hash){
print "$i\t$hash{$i}\n";
}


=abc
open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
@a=split "\t",$_;

}
close F2;

