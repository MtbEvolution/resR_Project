#!usr/bin/perl
use warnings;

my $n=0;
open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
if($n==0){
foreach $i(0..$#a){
#my @{$a[$i]};
$hash{$i}=$a[$i];
push @list, $a[$i];
}
$n++;
}else{
foreach $i(0..$#a){
if($a[$i] ne ""){
$a[$i]=~s/%//;
push @{$hash{$i}}, $a[$i];
}
}
}
}
close F1;

foreach $j(@list){
my $in1=0;
my $in2=0;
my $in3=0;
my $in4=0;
my $in5=0;
@b=@$j;
foreach $k(@b){
if($k>0 && $k<=20){
$in1++;
}elsif($k>20 && $k<=40){
$in2++;
}elsif($k>40 && $k<=60){
$in3++;
}elsif($k>60 && $k<=80){
$in4++;
}elsif($k>80 && $k<=100){
$in5++;
}
}
$fq1=$in1/($#b+1);
$fq2=$in2/($#b+1);
$fq3=$in3/($#b+1);
$fq4=$in4/($#b+1);
$fq5=$in5/($#b+1);
@c=($fq1,$fq2,$fq3,$fq4,$fq5);
$st=stdev(@c);
print "$j\t$st\n";
}


sub average{
my(@data) = @_;
if (not @data) {
die("Empty arrayn");
}
my $total = 0;
foreach $m(@data){
$total+=$m;
}
my $average = $total / @data;
return $average;
}
sub stdev{
my(@data) = @_;
if(@data == 1){
return 0;
}
my $average = &average(@data);
my $sqtotal = 0;
foreach(@data) {
$sqtotal += ($average-$_) ** 2;
}
my $std = ($sqtotal / (@data-1)) ** 0.5;
return $std;
}

