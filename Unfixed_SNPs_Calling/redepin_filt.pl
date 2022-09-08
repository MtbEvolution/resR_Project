#!usr/bin/perl
use warnings;

# repeat loci filter (twice and above, across different isolates)
# depth 0.5-1.5x ave depth
# ave reads location: 25%~75%
# duplicate reads location ratio: <50%

$name="$ARGV[2]";
$name=~s/info//;
$name1=$name."kept";
$name2=$name."disc";

open OUT1, ">$name1";
open OUT2, ">$name2";

open F0, $ARGV[0] or die $!;
while(<F0>){
chomp;
$hash{$_}=1;
}
close F0;

open F1, $ARGV[1] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
$dep=$a[2];
$dep=~s/ //;
$depup=$dep*1.5; # depth should not exceed 1.5-fold ave depth
$deplw=$dep*0.5; # depth should not below 0.5-fold ave depth
}
close F1;

open F2, $ARGV[2] or die $!;
while(<F2>){
chomp;
@a=split "\t",$_;
if(!exists $hash{$a[8]}){
if($a[12] > $deplw && $a[12] < $depup){
if($a[0] > 0.25 && $a[0] < 0.75){
if($a[4] =~ m/0.1/){
@b=split ":",$a[6];
if($b[1]<50 && $a[5] < 0.5){
print OUT1 "$_\n";
}elsif($b[1]>50){
print OUT1 "$_\n";
}else{
print OUT2 "Filter\t$_\n";
}
}
}
}else{
print OUT2 "Depth\t$_\n";
}
}else{
print OUT2 "Repeat\t$_\n";
}
}
close F2;
close OUT1;
close OUT2;

=pod
open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
}
close F2;
=cut

=pod
open F2, $ARGV[2] or die $!;
while(<F3>){
chomp;
}
close F3;
=cut
