# !/bin/bash

RESULTS_FOLDER="../data/results/composition"
SAVE_FOLDER="../plots/summary"

# dSprites
python -m analysis.summary --model_folders $RESULTS_FOLDER/{4,5,84,85} \
                           --name sqr2px \
                           --save $SAVE_FOLDER \
                           --all

# 3DShapes
python -m analysis.summary --model_folders $RESULTS_FOLDER/{13,14,92,93} \
                           --name shape2ohue \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/{23,24,98,99} \
                           --name shape2ohue-even-hues \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/{34,35,90,91} \
                           --name whue2fhue \
                           --save $SAVE_FOLDER \
                           --all

# MPI3D
python -m analysis.summary --model_folders $RESULTS_FOLDER/{76,77} \
                           --name shape2vx \
                           --save $SAVE_FOLDER \
                           --all

# Bad condition!!
python -m analysis.summary --model_folders $RESULTS_FOLDER/{78,79,80,81} \
                           --name cyl2bkg \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/{149,150} \
                           --name cyl2bkg \
                           --save $SAVE_FOLDER \
                           --all

# Circles
python -m analysis.summary --model_folders $RESULTS_FOLDER/117 \
                           --name circles-corner \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/118 \
                           --name circles-midpos \
                           --save $SAVE_FOLDER \
                           --all

# Simple
python -m analysis.summary --model_folders $RESULTS_FOLDER/122 \
                           --name simple-corner \
                           --save $SAVE_FOLDER \
                           --all

python -m analysis.summary --model_folders $RESULTS_FOLDER/128 \
                           --name simple-midpos \
                           --save $SAVE_FOLDER \
                           --all


# Supervised

RESULTS_FOLDER="../data/results/predictors"
python -m analysis.summary --model_folders $RESULTS_FOLDER/{1,15,13} \
                           --name supervised \
                           --save $SAVE_FOLDER \
                           --score --latent_reps
