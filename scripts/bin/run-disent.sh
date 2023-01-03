# !/bin/bash

# #======================== Dsprites ===================================================

# MODELS='abdi'
# BETAS='1 8 12'
# GAMMAS='20 50 100'

#  # Extrapolation
#  for MODELNAME in $MODELS
#  do
#      for B in $BETAS
#      do
#          python -m experiments.disent -n shapes3d with dataset.dsprites \
#                                          dataset.condition=extrp \
#                                          dataset.variant=blank_side \
#                                          model.$MODELNAME \
#                                          training.beta \
#                                          training.loss.params.beta=$B
#      done

#      for G in $GAMMAS
#      do
#          python -m experiments.disent -n shapes3d with dataset.dsprites \
#                                          dataset.condition=extrp \
#                                          dataset.variant=blank_side \
#                                          model.$MODELNAME \
#                                          training.factor \
#                                          training.loss.params.gamma=$G
#      done
#  done


# # Recombination to range
# SHAPES_R2R_VARIANTS='shape2tx'

# for V in $SHAPES_R2R_VARIANTS
# do
#     for MODELNAME in $MODELS
#     do
#         for B in $BETAS
#         do
#         python -m experiments.disent with dataset.dsprites \
#                                           dataset.condition=recomb2range \
#                                           dataset.variant=$V \
#                                           model.$MODELNAME \
#                                           training.beta \
#                                           training.loss.params.beta=$B
#          done

#      	for G in $GAMMAS
#         do
#             python -m experiments.disent with dataset.dsprites \
#                                               dataset.condition=recomb2range \
#                                               dataset.variant=$V \
#                                               model.$MODELNAME \
#                                               training.factor \
#                                               training.loss.params.gamma=$G
#         done
#     done
# done


# # Recombination to element
#  for MODELNAME in $MODELS
#  do
#      for B in $BETAS
#      do
#          python -m experiments.disent -n shapes3d with dataset.dsprites \
#                                          dataset.condition=recomb2element \
#                                          dataset.variant=leave1out \
#                                          model.$MODELNAME \
#                                          training.beta \
#                                          training.loss.params.beta=$B
#      done

#      for G in $GAMMAS
#      do
#          python -m experiments.disent -n shapes3d with dataset.dsprites \
#                                          dataset.condition=recomb2element \
#                                          dataset.variant=leave1out \
#                                          model.$MODELNAME \
#                                          training.factor \
#                                          training.loss.params.gamma=$G
#      done
# done


# #==================== SHAPES3D ==========================================

 # MODELS='kim'
 # BETAS='1 8 12'
 # GAMMAS='20 50 100'
 # GAMMAS='150 200'

 # Extrapolation
 # for MODELNAME in $MODELS
 # do
 #     for B in $BETAS
 #     do
 #         python -m experiments.disent -n shapes3d with dataset.shapes3d \
 #                                         dataset.condition=extrp \
 #                                         dataset.variant=fhue_gt50 \
 #                                         model.$MODELNAME \
 #                                         training.beta \
 #                                         training.loss.params.beta=$B
 #     done

 #     for G in $GAMMAS
 #     do
 #         python -m experiments.disent -n shapes3d with dataset.shapes3d \
 #                                         dataset.condition=extrp \
 #                                         dataset.variant=fhue_gt50 \
 #                                         model.$MODELNAME \
 #                                         training.factor \
 #                                         training.loss.params.gamma=$G
 #     done
 # done


# Recombination to range

 # SHAPES_R2R_VARIANTS='shape2fhue shape2ohue ohue2whue fhue2whue'
 # SHAPES_R2R_VARIANTS='fhue2whue'

 # for MODELNAME in $MODELS
 # do
 #     for B in $BETAS
 #     do
 #         for V in $SHAPES_R2R_VARIANTS
 #         do
 #         python -m experiments.disent -n shapes3d with dataset.shapes3d \
 #                                         dataset.condition=recomb2range \
 #                                         dataset.variant=$V \
 #                                         model.$MODELNAME \
 #                                         training.beta \
 #                                         training.loss.params.beta=$B
 #         done
 #     done

 #     for G in $GAMMAS
 #     do
 #         for V in $SHAPES_R2R_VARIANTS
 #         do
 #             python -m experiments.disent -n shapes3d with dataset.shapes3d \
 #                                             dataset.condition=recomb2range \
 #                                             dataset.variant=$V \
 #                                             model.$MODELNAME \
 #                                             training.factor \
 #                                             training.loss.params.gamma=$G

 #         done
 #     done
 # done

# Recombination to element
 # for MODELNAME in $MODELS
 # do
 #     for B in $BETAS
 #     do
 #         python -m experiments.disent -n shapes3d with dataset.shapes3d \
 #                                         dataset.condition=recomb2element \
 #                                         dataset.variant=leave1out \
 #                                         model.$MODELNAME \
 #                                         training.beta \
 #                                         training.loss.params.beta=$B
 #     done

 #     for G in $GAMMAS
 #     do
 #         python -m experiments.disent -n shapes3d with dataset.shapes3d \
 #                                         dataset.condition=recomb2element \
 #                                         dataset.variant=leave1out \
 #                                         model.$MODELNAME \
 #                                         training.factor \
 #                                         training.loss.params.gamma=$G
 #     done
 # done

#==================== MPI3D ==========================================


MODELS='montero abdi'
BETAS='1'
GAMMAS='20 50 100'
LAMBDA='10 20 30'


for MODELNAME in $MODELS
do
  for B in $BETAS
  do
    python -m experiments.disent with dataset.mpi3d \
                                      model.$MODELNAME \
                                      training.{beta,lr=0.0005} \
                                      training.loss.params.beta=$B
  done
done


for MODELNAME in $MODELS
do

  for L in $LAMBDA
  do
    python -m experiments.disent with dataset.mpi3d \
                                      model.$MODELNAME \
                                      training.{waemmd,lr=0.0001} \
                                      training.loss.params.lambda1=$L \
                                      training.loss.params.lambda2=0.001
  done
done


# for MODELNAME in $MODELS
# do
#   for G in $GAMMAS
#   do
#     python -m experiments.disent with dataset.mpi3d \
#                                       model.$MODELNAME \
#                                       training.{factor,lr=0.0001} \
#                                       training.loss.params.gamma=$G
#   done
# done


# # # Extrapolation
# MIP_EXTRP_VARIANTS='horz_gt20 objc_gt3'

# # for MODELNAME in $MODELS
# do
#     for B in $BETAS
#     do
#         for V in $MIP_EXTRP_VARIANTS
#         do
#         python -m experiments.disent -n mpi3d with dataset.mpi3d \
#                                         dataset.condition=extrp \
#                                         dataset.variant=$V \
#                                         model.$MODELNAME \
#                                         training.beta \
#                                         training.loss.params.beta=$B
#         done
#     done

#     for G in $GAMMAS
#     do
#         for V in $MIP_EXTRP_VARIANTS
#         do
#             python -m experiments.disent -n mpi3d with dataset.mpi3d \
#                                             dataset.condition=extrp \
#                                             dataset.variant=$V \
#                                             model.$MODELNAME \
#                                             training.factor \
#                                             training.loss.params.gamma=$G
#         done
#     done
# done


# # Recombination to range
# MPI_R2R_VARIANTS='cyl2horz objc2horz'

# for MODELNAME in $MODELS
# do
#     for B in $BETAS
#     do
#         for V in $MPI_R2R_VARIANTS
#         do
#             python -m experiments.disent -n mpi3d with dataset.mpi3d \
#                                             dataset.condition=recomb2range \
#                                             dataset.variant=$V \
#                                             model.$MODELNAME \
#                                             training.beta \
#                                             training.loss.params.beta=$B
#         done
#     done

#     for G in $GAMMAS
#     do
#         for V in $MPI_R2R_VARIANTS
#         do
#             python -m experiments.disent -n mpi3d with dataset.mpi3d \
#                                             dataset.condition=recomb2range \
#                                             dataset.variant=$V \
#                                             model.$MODELNAME \
#                                             training.factor \
#                                             training.loss.params.gamma=$G
#         done
#     done
# done

# # Recombination to element
# for MODELNAME in $MODELS
# do
#     for B in $BETAS
#     do
#         python -m experiments.disent -n mpi3d with dataset.mpi3d \
#                                         dataset.condition=recomb2element \
#                                         dataset.variant=leave1out \
#                                         model.$MODELNAME \
#                                         training.beta \
#                                         training.loss.params.beta=$B
#     done

#     for G in $GAMMAS
#     do
#         python -m experiments.disent -n mpi3d with dataset.mpi3d \
#                                         dataset.condition=recomb2element \
#                                         dataset.variant=leave1out \
#                                         model.$MODELNAME \
#                                         training.factor \
#                                         training.loss.params.gamma=$G
    # done
# done
