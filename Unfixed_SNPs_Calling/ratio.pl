#!usr/bin/perl
use warnings;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
$rat=$a[3]/$a[4];
print "$_\t$rat\n";
}
close F1;

=abc
open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
@a=split "\t",$_;

}
close F2;

