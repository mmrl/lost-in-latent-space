# !/bin/bash

# dSprites

MODELS='abdi'
# BETAS='1 8 12'
# GAMMAS='20 50 100'
BASELINE_ID='39 40 41 42 43 44'

SHAPES_R2R_VARIANTS='shape2tx'

for MODELNAME in $MODELS
do
    for D in $BASELINE_ID
    do
        for V in $SHAPES_R2R_VARIANTS
        do
        python -m experiments.finetune with dataset.{dsprites,condition=recomb2range,variant=$V} \
                                             model.{$MODELNAME,pretrained="../data/sims/disent/baseline/$D/",finetune_decoder=False,latent_size=10} \
                                             training.{vae,loss.params.beta=1,epochs=66}
        done
    done
done

# 3DShapes

MODELS='abdi'
BASELINE_ID='45 46 47 48 49 50'
SHAPES_R2R_VARIANTS='shape2ohue'

for MODELNAME in $MODELS
do
    for D in $BASELINE_ID
    do
        for V in $SHAPES_R2R_VARIANTS
        do
        python -m experiments.finetune with dataset.{shapes3d,condition=recomb2range,variant=$V} \
                                             model.{$MODELNAME,pretrained="../data/sims/disent/baseline/$D/",finetune_decoder=False,latent_size=10} \
                                             training.{vae,loss.params.beta=1,epochs=66}
        done
    done
done
