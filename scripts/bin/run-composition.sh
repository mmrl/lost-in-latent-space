# !/bin/bash

# dSprites

# all data
# python -m experiments.composition with dataset.dsprites \
# 				       model.{abdi,composition_op='linear'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.dsprites \
# 				       model.{abdi,composition_op='interp'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.dsprites \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.dsprites \
# 				       model.{abdi,composition_op='interp'} \
#                training.{waemmd,batch_size=64,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.dsprites \
# 				       model.{abdi,composition_op='fixint'} \
#                training.{waemmd,batch_size=64,lr=0.0001,epochs=100}

# recomb2range
# python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx} \
# 				       model.{abdi,composition_op='linear'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx} \
# 				       model.{abdi,composition_op='interp'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx} \
# 				       model.{abdi,composition_op='fixint'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# 3DShapes

# All data

# python -m experiments.composition with dataset.shapes3d \
# 				       model.{abdi,composition_op='interp'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       model.{abdi,composition_op='interp'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       model.{abdi,composition_op='fixint'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       model.{abdi,sbd4,composition_op='interp'} \
#                training.{lr=0.0003,epochs=20,batch_size=16}

# excluding colors
# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.modifiers='["half_ohues"]' \
# 				       model.{abdi,composition_op='interp'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.modifiers='["half_ohues"]' \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.modifiers='["half_ohues"]' \
# 				       model.{abdi,composition_op='interp'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.modifiers='["half_ohues"]' \
# 				       model.{abdi,composition_op='fixint'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# Recombination to Range

# floor and wall hue

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=fhue2whue} \
# 				       model.{abdi,composition_op='linear'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=fhue2whue} \
# 				       model.{abdi,composition_op='interp'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=fhue2whue} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=fhue2whue} \
# 				       model.{abdi,composition_op='interp'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=fhue2whue} \
# 				       model.{abdi,composition_op='fixint'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#  				       dataset.{condition=recomb2range,variant=fhue2whue} \
# 				       model.{abdi,sbd4,composition_op='interp'} \
#                training.{lr=0.0003,epochs=20,batch_size=16}


# shape and object hue

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2ohue} \
# 				       model.{abdi,composition_op='linear'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2ohue} \
# 				       model.{abdi,composition_op='interp'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2ohue} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2ohue} \
# 				       model.{abdi,composition_op='interp'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2ohue} \
# 				       model.{abdi,composition_op='fixint'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#  				       dataset.{condition=recomb2range,variant=shape2ohue} \
# 				       model.{abdi,sbd4,composition_op='interp'} \
#                training.{lr=0.0003,epochs=20,batch_size=16}


# shape to orientation

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2orient} \
# 				       model.{abdi,composition_op='interp'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2orient} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2orient} \
# 				       model.{abdi,composition_op='interp'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2orient} \
# 				       model.{abdi,composition_op='fixint'} \
#                training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#  				       dataset.{condition=recomb2range,variant=shape2orient} \
# 				       model.{abdi,sbd4,composition_op='interp'} \
#                training.{lr=0.0003,epochs=20,batch_size=16}


# with half th colors

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohueq} \
#               dataset.modifiers='["half_ohues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohueq} \
#               dataset.modifiers='["half_ohues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohueq} \
#               dataset.modifiers='["half_ohues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# with even colors

# floor and wall hue

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=fhue2whue} \
#               dataset.modifiers='["even_wnf_hues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=fhue2whue} \
#               dataset.modifiers='["even_wnf_hues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=fhue2whue} \
#               dataset.modifiers='["even_wnf_hues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=fhue2whue} \
#               dataset.modifiers='["even_wnf_hues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# shape and object hue

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# Recombination to element

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=leave1out} \
#               model.{abdi,composition_op=interp} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=leave1out} \
#               model.{abdi,composition_op=fixint} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=leave1out} \
#               model.{abdi,composition_op=interp} \
#               training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=leave1out} \
#               model.{abdi,composition_op=fixint} \
#               training.{waemmd,lr=0.0001,epochs=100}

# wiht odd colors

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# # with half the colors

 # python -m experiments.composition with dataset.shapes3d \
 #              dataset.{condition=recomb2element,variant=shape2ohue} \
 #              dataset.modifiers='["half_ohues"]' \
 #              model.{abdi,composition_op='interp'} \
 #              training.{lr=0.0001,epochs=100}

 # python -m experiments.composition with dataset.shapes3d \
 #              dataset.{condition=recomb2element,variant=shape2ohue} \
 #              dataset.modifiers='["half_ohues"]' \
 #              model.{abdi,composition_op='fixint'} \
 #              training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=shape2ohue} \
#               dataset.modifiers='["half_ohues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2element,variant=shape2ohue} \
#               dataset.modifiers='["half_ohues"]' \
#               model.{abdi,composition_op='fixint'} \
#               training.{waemmd,lr=0.0001,epochs=100}

# MPI dataset

# all data

# Using VAEs
# python -m experiments.composition with dataset.mpi3d \
#   dataset.modifiers='["four_shapes"]' \
#   model.{montero,composition_op='interp'} \
#   training.{lr=0.0001,epochs=50}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.modifiers='["four_shapes"]' \
#   model.{montero,composition_op='fixint'} \
#   training.{lr=0.0001,epochs=50}

# cylinder vs vx

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2hx} \
#   model.{montero,composition_op='interp'} \
#   training.{lr=0.0001,epochs=50}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2hx} \
#   model.{montero,composition_op='fixint'} \
#   training.{lr=0.0001,epochs=50}

# using WAE
# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2hx} \
#   model.{montero,composition_op='interp'} \
#   training.{waemmd,lr=0.0001,epochs=50}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2hx} \
#   model.{montero,composition_op='fixint'} \
#   training.{waemmd,lr=0.0001,epochs=50}

# # cylinder vs vertical axis (half of horizontal axis)

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "lhalf_hx"]' \
#   model.{abdi,composition_op='interp'} \
#   training.{lr=0.0001,epochs=50}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "lhalf_hx"]' \
#   model.{abdi,composition_op='fixint'} \
#   training.{lr=0.0001,epochs=50}

# # using WAE

# python -m experiments.composition with dataset.mpi3d \
#   dataset.modifiers='["four_shapes", "lhalf_hx"]' \
#   model.{montero,composition_op='interp'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.modifiers='["four_shapes", "lhalf_hx"]' \
#   model.{montero,composition_op='fixint'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.modifiers='["four_shapes", "lhalf_hx"]' \
#   model.{montero,composition_op='fixint'} \
#   training.{waemmd,batch_size=128,lr=0.0001,epochs=200}

# recomb2range

# cylinder vs vx (fixed horizontal axis and only four shapes)

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,composition_op='interp'} \
#   training.{lr=0.0001,epochs=50}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,composition_op='fixint'} \
#   training.{lr=0.0001,epochs=50}


# using WAE

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,composition_op='interp'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200} \
#   training.rebalance_dataset=True

python -m experiments.composition with dataset.mpi3d \
  dataset.{condition=recomb2range,variant=cyl2vx} \
  dataset.modifiers='["four_shapes", "fix_hx"]' \
  model.{montero,composition_op='fixint'} \
  training.{waemmd,batch_size=64,lr=0.0001,epochs=200} \
  training.rebalance_dataset=True

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,sbd4,composition_op='interp'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,sbd4,composition_op='fixint'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# # cylinder vs background (fixed horizontal axis)

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=bkg2cyl} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,composition_op='interp'} \
#   training.{lr=0.0001,epochs=100}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=bkg2cyl} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,composition_op='fixint'} \
#   training.{lr=0.0001,epochs=100}

# using WAE

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=bkg2cyl} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,composition_op='interp'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=bkg2cyl} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,composition_op='fixint'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=bkg2cyl} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,sbd4,composition_op='interp'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=bkg2cyl} \
#   dataset.modifiers='["four_shapes", "fix_hx"]' \
#   model.{montero,sbd4,composition_op='fixint'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# # cylinder vs vertical axis (half of horizontal axis)

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "lhalf_hx"]' \
#   model.{montero,composition_op='interp'} \
#   training.{lr=0.0001,epochs=50}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "lhalf_hx"]' \

#   training.{lr=0.0001,epochs=50}

# cylinder vs vertical axis (even vertical axis values)

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "fix_hx", "even_vx"]' \
#   model.{montero,composition_op='fixint'} \
#   training.{waemmd,batch_size=64,lr=0.0001,epochs=200}

# python -m experiments.composition with dataset.mpi3d \
#   dataset.{condition=recomb2range,variant=cyl2vx} \
#   dataset.modifiers='["four_shapes", "lhalf_hx", "even_vx"]' \
#   model.{montero,composition_op='fixint'} \
