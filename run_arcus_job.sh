#!/bin/bash
ssh login11
cd $DATA
now=$(date +"%m_%d_%Y")
dir=${DATA}/pints_matrix_${now}
rm -rf $dir
mkdir $dir
cd $dir
module load git
git clone git@github.com:martinjrobins/2017PintsPaper.git
cd 2017PintsPaper
git clone https://github.com/pints-team/pints.git pints_repo
mv pints_repo/pints .
module load python/3.5
pip install --target=. gpyopt
s=`python3 main.py --max`
sbatch --array=0-$s arcus_job.sh
