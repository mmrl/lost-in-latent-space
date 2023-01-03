# # !/bin/bash

COMPOSITION=../data/sims/composition
SUPERVISED=../data/sims/supervised

############################ Failure conditions ###############################

# dSprites

MODEL_IDS="4 5 84 85"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.{factor1=shape,factor2=posX} \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done

# Shapes3D

MODEL_IDS="13 14 92 93"
for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.{factor1=shape,factor2=object_hue} \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done

MODEL_IDS="23 24 98 99"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.{factor1=shape,factor2=object_hue} \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done


MODEL_IDS="158 159"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.{factor1=shape,factor2=orientation} \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done


# MPI3D

MODEL_IDS="74 76 77"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.factor1=object_shape \
                           analysis.factor2=vertical_axis \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done

############################# Success conditions ###############################

# Shapes3D

MODEL_IDS="34 35 90 91"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.{factor1=floor_hue,factor2=wall_hue} \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done

# MPI3D

# MODEL_IDS="78 79 80 81" # Only one background color excluded here!!!

# for ID in $MODEL_IDS
# do
#   python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
#                            analysis.factor1=background_color \
#                            analysis.factor2=object_shape \
#                            analysis.n_recons=5 \
#                            analysis.method_args="{'precompute': False,
#                                                   'max_iter': 10000}"
# done


MODEL_IDS="149 150" # Two background colors excluded. Version used in main text

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.factor1=background_color \
                           analysis.factor2=object_shape \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done


### SBD

# Circles

MODEL_IDS="117 118"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.{factor1=y,factor2=x} \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done

# Simple

MODEL_IDS="122 128"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$COMPOSITION model_id=$ID \
                           analysis.{factor1=shape,factor2=x} \
                           analysis.n_recons=5 \
                           analysis.method_args="{'precompute': False,
                                                  'max_iter': 10000}"
done


############################### Supervised #####################################

# dSprites

MODEL_IDS="1"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$SUPERVISED model_id=$ID \
                           analysis.{factor1=shape,factor2=posX}
done

# Shapes3D

MODEL_IDS="15"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$SUPERVISED model_id=$ID \
                           analysis.{factor1=shape,factor2=object_hue}
done

# MPI3D

MODEL_IDS="13"

for ID in $MODEL_IDS
do
  python -m analysis.model with exp_folder=$SUPERVISED model_id=$ID \
                           analysis.{factor1=object_shape,factor2=vertical_axis}
