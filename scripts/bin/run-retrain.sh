# !/bin/bash

# dSprites

python -m experiments.finetune with dataset.dsprites \
            dataset.{condition=recomb2range,variant=sqr2tx} \
            model.{abdi,composition_op='fixint',retrain_decoder=False} \
            model.base_model="../data/sims/composition/15/" \
            training.{lr=0.0001,epochs=100} \
            training.rebalance_dataset=True

python -m experiments.finetune with dataset.dsprites \
            dataset.{condition=recomb2range,variant=sqr2tx} \
            model.{abdi,composition_op='fixint',retrain_decoder=False} \
            model.base_model="../data/sims/composition/15/" \
            training.{lr=0.0001,epochs=100} \
            training.rebalance_dataset=True

# Shapes3D

python -m experiments.finetune with dataset.shapes3d \
            dataset.{condition=recomb2range,variant=shape2ohue} \
            model.{abdi,composition_op='interp',retrain_decoder=True} \
            model.base_model="../data/sims/composition/20/" \
            training.{lr=0.0001,epochs=100} \
            training.rebalance_dataset=True

python -m experiments.finetune with dataset.shapes3d \
            dataset.{condition=recomb2range,variant=shape2ohue} \
            model.{abdi,composition_op='interp',retrain_decoder=True} \
            model.base_model="../data/sims/composition/20/" \
            training.{lr=0.0001,epochs=100} \
            training.rebalance_dataset=True
