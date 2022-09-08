#!usr/bin/perl
use warnings;

# 1) filter those "Pass value" != "1E0";
# 2) filter symbols "$, ^", filter INDELs;

my $k=0;

open F1, $ARGV[0] or die $!;
while(<F1>){
chomp;
@a=split "\t",$_;
if($a[8] =~m/Pass=1E0/){  #filter those variants with varscan P value failed;
$a[13]=~s/\$//g; # filter those "$"
$a[13]=~s/\^.//g; # filter those "^"
$len=length $a[13];
foreach $i(0..$len-1){
	$a=substr $a[13],$i,1;
	if($a =~ m/\+|\-/){
		$i1=$i+1;
		$i2=$i+2;
		$a1=substr $a[13],$i1,1;
		$a2=substr $a[13],$i2,1;
		if($a1 =~ m/\d/ && $a2 !~ m/\d/){
			$k=$a1+1; # use $k to indicate how many bases should be skipped
		}elsif($a1 =~ m/\d/ && $a2 =~ m/\d/){
			$k=$a1*10+$a2+2;
		}
	}elsif($a !~ m/\+|\-/ && $k>0){
		$k--;   # each time skip a base, $k self-minus
	}elsif($a !~ m/\+|\-/ && $k==0){
		$seq.="$a"; # when $k==0, print non-indel bases
	}
}
$len1=length $seq;
@b=split ",",$a[14];
$len2=$#b+1;
if($len1==$len2){ # the length of base line and position line should be the same
print "$a[0]\t$a[1]\t$a[2]\t$a[3]\t$a[4]\t$a[5]\t$a[6]\t$a[7]\t$a[8]\t$a[9]\t$a[10]\t$a[11]\t$a[12]\t$seq\t$a[14]\n";
}else{
print "ERROR$a[0]\t$a[1]\t$a[2]\t$a[3]\t$a[4]\t$a[5]\t$a[6]\t$a[7]\t$a[8]\t$a[9]\t$a[10]\t$a[11]\t$a[12]\t$seq\t$a[14]\n";
}
$seq="";
}
}
close F1;
