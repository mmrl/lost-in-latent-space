# !/bin/bash

RESULTS_FOLDER="../data/results/composition"
SAVE_FOLDER="../plots/figures/"

# Failures figure

DSPRITES=5
SHAPES3D=23
MPI3D=77

python -m analysis.condition --model_folders $RESULTS_FOLDER/{5,23,77} \
                             --name "failure-cases.pdf" \
                             --save $SAVE_FOLDER \
                             --plot_recons


# Successes figure

SHAPES3D=90
MPI3D=150

python -m analysis.condition --model_folders $RESULTS_FOLDER/{90,150} \
                             --name "success-cases.pdf" \
                             --save $SAVE_FOLDER \
                             --plot_recons


# SBD Figure

CIRCLES=118
SIMPLE=128

python -m analysis.condition --name "circles-simple-midpos.pdf" \
                             --model_folders $RESULTS_FOLDER/{118,128} \
                             --save $SAVE_FOLDER \
                             --plot_recons

# Supervised models

RESULTS_FOLDER="../data/results/supervised"

DSPRITES=1
SHAPES3D=15
MPI3D=13

python -m analysis.condition --name "supervised-failures.pdf" \
                             --model_folders $RESULTS_FOLDER/{1,15,13} \
                             --save ../data/results/conditions/
