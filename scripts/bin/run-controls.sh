# !/bin/bash


################################### dSprites ##################################

# rebalance datasset
# python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx} \
# 				       model.{abdi,composition_op='interp'} \
# 				       training.{lr=0.0001,epochs=100} \
#                training.rebalance_dataset=True

# python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100} \
#                training.rebalance_dataset=True


# Shifting generalisation window

# CUDA_VISIBLE_DEVICES="1" python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx_rs} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100} \
#                training.rebalance_dataset=True &> /dev/null &

# CUDA_VISIBLE_DEVICES="2" python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx_cent} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100} \
#                training.rebalance_dataset=True &> /dev/null &

# CUDA_VISIBLE_DEVICES="3" python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2tx_flnk} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100} \
#                training.rebalance_dataset=True &> /dev/null &


# Testing shape vs scale (with rebalancing since it seems to improve generalisation)

# CUDA_VISIBLE_DEVICES="2" python -m experiments.composition with dataset.dsprites \
# 				       dataset.{condition=recomb2range,variant=sqr2scl} \
# 				       model.{abdi,composition_op='fixint'} \
# 				       training.{lr=0.0001,epochs=100} \
#                training.rebalance_dataset=True &> /dev/null &

# wait

################################### 3DShapes ##################################

# shape and object Hue
# CUDA_VISIBLE_DEVICES="1" python -m experiments.composition with dataset.shapes3d \
#               dataset.{condition=recomb2range,variant=shape2ohue} \
#               dataset.modifiers='["even_ohues"]' \
#               model.{abdi,composition_op='interp'} \
#               training.{lr=0.0001,epochs=100} \
#               training.rebalance_dataset=True &> /dev/null &


# CUDA_VISIBLE_DEVICES="2" python -m experiments.composition with dataset.shapes3d \
# 				      dataset.{condition=recomb2range,variant=shape2orient} \
# 				      model.{abdi,composition_op='interp'} \
# 				      training.{lr=0.0001,epochs=100} \
#               training.rebalance_dataset=True &> /dev/null &

CUDA_VISIBLE_DEVICES="3" python -m experiments.composition with dataset.shapes3d \
				      dataset.{condition=recomb2range,variant=shape2ohue_flnk} \
				      model.{abdi,composition_op='interp'} \
				      training.{lr=0.0001,epochs=100} \
              training.rebalance_dataset=True &> /dev/null &

wait


################################## MPI3D #######################################

# CUDA_VISIBLE_DEVICES="3" python -m experiments.composition with dataset.mpi3d \
#               dataset.{condition=recomb2range,variant=cyl2vx} \
#               dataset.modifiers='["four_shapes", "fix_hx"]' \
#               model.{montero,composition_op='fixint'} \
#               training.{waemmd,batch_size=64,lr=0.0001,epochs=200} \
#               training.rebalance_dataset=True &> /dev/null &
# wait
