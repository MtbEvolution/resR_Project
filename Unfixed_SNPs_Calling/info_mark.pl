#!usr/bin/perl
use warnings;

my $k=0;
my @location;
my $readlen=0;



open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
$len=length $a[13];
@b=split ",",$a[14];
foreach $i(@b){
if($i>$readlen){
$readlen=$i;
}
}
foreach $j(0..$len-1){
$nu=substr $a[13], $j,1;
if($nu=~m/A|T|G|C|a|t|g|c/){
push @location,$b[$j];
$loca+=$b[$j];
$med=sprintf "%.0f",$len/2;
$dis=abs($j-$med);
$sum+=$dis;
$num++;
}
}
$aveloca=sprintf "%.0f",$loca/$num;
$ratioloca=sprintf "%.2f",$aveloca/$readlen;
$avedis=sprintf "%.0f",$sum/$num;
$ratiodis=abs(sprintf "%.2f",$avedis/$med);

sub chi_squared {
     my ($a,$b,$c,$d) = @_;
     return 0 if($b+$d == 0);
     my $n= $a + $b + $c + $d;
     return (($n*($a*$d - $b*$c)**2) / (($a + $b)*($c + $d)*($a + $c)*($b + $d)));
}
@c1=split "=",$a[6];
@c2=split ":",$c1[1];
@d1=split "=",$a[7];
@d2=split ":",$d1[1];
@chis=($c2[0],$c2[1],$d2[0],$d2[1]);
$p="NA";
if($c2[0]+$c2[1]!=0 && $d2[0]+$d2[1]!=0 && $c2[0]+$d2[0]!=0 && $c2[1]+$d2[1]!=0){
$chi=chi_squared(@chis);
if($chi<=2.71){
$p=">=0.1";
}elsif($chi>2.71 && $chi <= 3.84){
$p="0.05~0.1";
}elsif($chi>3.84 && $chi <= 6.63){
$p="0.01~0.05";
}elsif($chi>6.63 && $chi <= 7.88){
$p="0.005~0.01";
}elsif($chi> 7.88){
$p="<0.005";
}
}

my $all=0;
foreach $m(@location){
if(!exists $hash1{$m}){
$hash1{$m}=1;
}else{
if(!exists $hash2{$m}){
$hash2{$m}=2;
}else{
$hash2{$m}++;
}
}
}
foreach $n(keys %hash2){
$all+=$hash2{$n};
}
$per=sprintf "%.2f", $all/$d1[0];

print "$ratioloca\t$readlen:$aveloca\t$ratiodis\t$med:$avedis\t$p\t$per\t$all:$d1[0]\t$_\t@location\n";
$loca=0;
@location="";
$sum=0;
$num=0;
%hash1=();
%hash2=();
}

close F1;
