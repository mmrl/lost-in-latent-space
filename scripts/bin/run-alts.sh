# !/bin/bash

# dSprites

# all data
# for i in {1..5}
# do
#   python -m experiments.disent with dataset.dsprites \
#                  model.cascade \
#                  training.cascade
# done

# for i in {1..5}
# do
#   python -m experiments.disent with dataset.dsprites \
#                  dataset.{condition=recomb2range,variant=sqr2tx} \
#                  model.cascade training.{cascade,epochs=40,step_lr}
# done

# for i in {1..5}
# do
#   python -m experiments.disent with dataset.dsprites \
#                  dataset.{condition=recomb2range,variant=sqr2tx} \
#                  model.cascade training.{cascade,epochs=40,step_lr} \
#                  training.rebalance_dataset=True
# done

# 3DShapes

# all data
# python -m experiments.disent with dataset.shapes3d \
# 				       model.{montero,cascade,n_cat=4,lmbda=0.0001} \
# 				       training.{cascade,epochs=60,lr=0.0001} -u

# floor and wall hue
# python -m experiments.disent with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=fhue2whue} \
# 				       model.{cascade,n_cat=4,lmbda=0.1} \
# 				       training.{cascade,epochs=60,lr=0.0001}

# shape and object hue
# python -m experiments.disent with dataset.shapes3d \
# 				       dataset.{condition=recomb2range,variant=shape2ohue} \
# 				       model.{cascade,n_cat=4} \
# 				       training.{cascade,epochs=60}


# MPI dataset
# python -m experiments.disent with dataset.mpi3d \
#                dataset.modifiers='["four_shapes, fix_hx"]' \
# 				       model.{cascade,n_cat=4} \
# 				       training.{cascade,epochs=60}

# cylinder vs vx
# python -m experiments.disent with dataset.mpi3d \
#                dataset.modifiers='["four_shapes", "fix_hx"]' \
#                dataset.{condition=recomb2range,variant=cyl2hx} \
# 				       model.{cascade,n_cat=4} \
# 				       training.{cascade,epochs=60}

# cylinder vs background (fixed horizontal axis)
# python -m experiments.disent with dataset.mpi3d \
#                dataset.modifiers='["four_shapes", "fix_hx"]' \
#                dataset.{condition=recomb2range,variant=bkg2cyl} \
# 				       model.{cascade,n_cat=4} \
# 				       training.{cascade,epochs=60}



################################## LieGroupVAE ################################

# dSprites (all data)
# for i in {1..5}
# do
#   python -m experiments.disent with dataset.dsprites \
#                   model.liegroup  \
#                   training.{lievae,epochs=50}
# done

# dSprites (generalisation)
# for i in {1..5}
# do
#   python -m experiments.disent with dataset.dsprites \
#                  dataset.{condition=recomb2range,variant=sqr2tx} \
#                  model.liegroup \
#                  training.{lievae,epochs=100,rebalance_dataset=True}
# done


# 3DShapes

# all data
# for i in {1..5}
# do
#   python -m experiments.disent with dataset.shapes3d \
#                  model.{liegroup,latent_size="[[400],[10]]"} \
#                  training.{lievae,epochs=50} \
#                  training.loss.params.hy_hes=20 \
#                  training.loss.params.subgroup_sizes="[400]"
# done

# floor and wall hue
for i in {1..4}
do
  python -m experiments.disent with dataset.shapes3d \
                 dataset.{condition=recomb2range,variant=fhue2whue} \
                 model.{liegroup,latent_size="[[400],[10]]"} \
                 training.{lievae,epochs=100,rebalance_dataset=True} \
                 training.loss.params.{hy_hes=20,subgroup_sizes="[400]"}
done

# # shape and object hue
for i in {1..4}
do
  python -m experiments.disent with dataset.shapes3d \
                 dataset.{condition=recomb2range,variant=shape2ohue} \
                 model.{liegroup,latent_size="[[400],[10]]"} \
                 training.{lievae,epochs=100,rebalance_dataset=True} \
                 training.loss.params.{hy_hes=20,subgroup_sizes="[400]"}
done
