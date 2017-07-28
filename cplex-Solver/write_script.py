towrite = """#!/bin/bash -l
#
#$ -cwd
#$ -j y
#$ -q idra
#$ -o out/cplex-Solver-%s-%s-unlimit.out
#$ -m e
#$ -M nguyentaduy@gmail.com
module load anaconda
module load cplex-studio/12.7.0.0.0
source activate python3.5
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python cplex-Solver-%s.py
"""
for i in range(10):
    f = "qsub-script-" + str(i)
    with open(f, "w") as f:
        f.write(towrite % (str(i), "full", str(i)))
