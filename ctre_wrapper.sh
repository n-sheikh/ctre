#!/bin/bash -l
#$ -N testfw
#$ -l h_vmem=120G
#$ -l 'GPU=1'
#$ -j y
#$ -cwd

module load anaconda/3.2019.10/default
module load cuda/10.1/default
source activate ctre

# cd to the local scratch storage and create “input” and “results” sub-directories
echo $SGE_O_WORKDIR
echo $TMPDIR

echo "$(pwd)"
cd $TMPDIR
echo "$(pwd)"


# mkdir output
# echo "$(ls)"
# copy your input data into “input” directory on scratch storage

# cd $SGE_O_WORKDIR
echo "$(pwd)"
cp -r $SGE_O_WORKDIR .
# cp -r $SGE_O_WORKDIR/i/ 
cd ctre
echo "$(pwd)"
echo "$(ls)"
python $TMPDIR/ctre/processing-resources/dl-framework/main.py $TMPDIR/ctre config 
cp -r $TMPDIR/ctre/cnc-task-3/experiments  $SGE_O_WORKDIR/cnc-task-3/
cp -r $TMPDIR/ctre/cnc-task-3/script-development/output/ $SGE_O_WORKDIR/cnc-task-3/script-development/
