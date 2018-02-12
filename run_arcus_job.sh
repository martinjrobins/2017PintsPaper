#!/bin/bash
cd $DATA
now=$(date +"%m_%d_%Y")
dir=pints_matrix_$now
rm -rf $dir
mkdir $dir
cd $dir
git clone https://github.com/pints-team/pints.git
cd pints
module load python
pip install --user .
cd ..
git clone git@github.com:martinjrobins/2017PintsPaper.git 
cd 2017PintsPaper
s=`python main.py --size`
echo $s > test.out