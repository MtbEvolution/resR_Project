#!usr/bin/perl
use warnings;

my %hash;
my %info;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
$hash{$a[1]}=1;
$info{$a[1]}=$_;
}
close F1;

open F2, $ARGV[1] or die $!;
while(<F2>){
chomp;
@b=split "\t",$_;
if(exists $hash{$b[1]}){
print "$info{$b[1]}\t$b[4]\t$b[6]\t$b[5]\n";
}
}
close F2;
