# !/bin/bash


# dSprites

# python -m experiments.supervised with model.kim \
# 				      dataset.{dsprites,condition=recomb2range,variant=ell2tx} \
# 				      training.regression

# python -m experiments.supervised with model.kim \
# 				      dataset.{dsprites,modifiers=["sparse_posX"]} \
#               dataset.{condition=recomb2range,variant=ell2tx} \
# 				      training.regression


# 3DShapes

# All data simulations

# python -m experiments.supervised with model.kim \
# 				      dataset.{shapes3d,pred_type="reg"} \ # reg is the default
# 				      training.regression

# python -m experiments.supervised with model.kim \
# 				      dataset.{shapes3d,pred_type="class",decision_dim=4} \
# 				      training.multiclass

# OOD condition

# python -m experiments.supervised with model.kim \
# 				      dataset.{shapes3d,condition=recomb2range,variant=shape2ohue} \
# 				      training.regression

python -m experiments.supervised with model.kim \
				      dataset.{shapes3d,modifiers="['even_ohues']"} \
              dataset.{condition=recomb2range,variant=shape2ohue} \
				      training.regression seed=148645996

# python -m experiments.supervised with model.kim \
# 				      dataset.{shapes3d,condition=recomb2range,variant=shape2fhue} \
# 				      training.regression

# python -m experiments.supervised with model.kim \
# 				      dataset.{shapes3d,condition=recomb2range,variant=ohue2whue} \
# 				      training.regression

# python -m experiments.supervised with model.kim \
# 				      dataset.{shapes3d,condition=recomb2range,variant=fhue2whue} \
# 				      training.regression


# MPI3D

# All data

# python -m experiments.supervised with model.montero \
#           dataset.{mpi3d,norm_lats=True,modifiers='["four_shapes"]'} \
#           dataset.{condition=recomb2range,variant=cyl2vx} \
#           training.regression

# Restrict vx

# python -m experiments.supervised with model.montero \
#           dataset.{mpi3d,norm_lats=True,modifiers='["four_shapes", "fix_hx"]'} \
#           training.regression

# Recomb2range

# python -m experiments.supervised with model.montero \
#           dataset.{mpi3d,norm_lats=True,modifiers='["four_shapes", "fix_hx"]'} \
#           dataset.{condition=recomb2range,variant=cyl2vx} \
#           training.regression

# python -m experiments.supervised with model.montero \
#           dataset.{mpi3d,norm_lats=True,modifiers='["four_shapes", "fix_hx"]'} \
#           dataset.{condition=recomb2range,variant=bkg2cyl} \
#           training.regression
