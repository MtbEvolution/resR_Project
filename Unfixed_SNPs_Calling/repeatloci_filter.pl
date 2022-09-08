#!usr/bin/perl
use warnings;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
#@a=split "\t",$_;
$hash{$_}=1;
}
close F1;

open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
@a=split "\t",$_;
if(!exists $hash{$a[8]}){
print "$_\n";
}
}
close F2;

