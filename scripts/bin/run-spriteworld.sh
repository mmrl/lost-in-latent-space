# !/bin/bash

# Circles shapes

# python -m experiments.composition with \
#              dataset.circles \
#              model.{kim,sbd,compositional_op='fixint'} \
#              training.{lr=0.0003,epochs=100,batch_size=16}

# python -m experiments.composition with \
#              dataset.circles \
#              dataset.{condition=recomb2range,variant=shape2px} \
#              model.{kim,sbd,compositional_op='fixint'} \
#              training.{lr=0.0003,epochs=100,batch_size=16}

# python -m experiments.composition with \
#              dataset.circles \
#              dataset.{condition=recomb2range,variant=midpos} \
#              model.{kim,sbd,compositional_op='fixint'} \
#              training.{lr=0.0003,epochs=100,batch_size=16}


# Simple shapes

# python -m experiments.composition with \
#             dataset.simple\
#             model.{kim,compositional_op='fixint'} \
#             training.{lr=0.0001,epochs=100,batch_size=16}

# python -m experiments.composition with \
#             dataset.simple\
#             model.{kim,sbd,compositional_op='fixint'} \
#             training.{lr=0.0001,epochs=100,batch_size=16}

# python -m experiments.composition with \
#             dataset.{simple,condition=recomb2range,variant=shape2px} \
#             model.{kim,compositional_op='fixint'} \
#             training.{lr=0.0001,epochs=100,batch_size=16}

# python -m experiments.composition with \
#             dataset.{simple,condition=recomb2range,variant=shape2px} \
#             model.{kim,sbd,compositional_op='fixint'} \
#             training.{lr=0.0001,epochs=100,batch_size=16}

# python -m experiments.composition with \
#             dataset.{simple,condition=recomb2range,variant=midpos} \
#             model.{kim,sbd,compositional_op='fixint'} \
#             training.{lr=0.0001,epochs=100,batch_size=16}

# Two shapes

# python -m experiments.composition with \
#             dataset.twoshapes \
#             model.{kim,sbd,compositional_op='interp'} \
#             training.{lr=0.0001,epochs=100}

# python -m experiments.composition with \
#             dataset.twoshapes \
#             model.{kim,sbd,compositional_op='fixint'} \
#             training.{lr=0.0001,epochs=100}

# recombination to range

# python -m experiments.composition with \
#             dataset.{twoshapes,modifiers='["uniside"]'} \
#             dataset.{condition=recomb2range,variant=indephue} \
#             model.{kim,compositional_op='interp'} \
#             training.{lr=0.0001,epochs=100}

# python -m experiments.composition with \
#             dataset.twoshapes \
#             dataset.{condition=recomb2range,variant=huecomb} \
#             model.{kim,sbd,compositional_op='interp'} \
#             training.{lr=0.0001,epochs=100}

# python -m experiments.composition with \
#             dataset.twoshapes \
#             dataset.{condition=recomb2range,variant=huecomb} \
#             model.{kim,sbd,compositional_op='fixint'} \
#             training.{lr=0.0001,epochs=100}


# All possible non-occluded positions

# python -m experiments.composition with \
#             dataset.{twoshapes,modifiers='["noverlap"]'} \
#             model.{kim,sbd,compositional_op='interp'} \
#             training.{lr=0.0003,epochs=20,batch_size=16}

# python -m experiments.composition with \
#             dataset.{twoshapes,modifiers='["noverlap"]'} \
#             model.{kim,sbd,compositional_op='fixint'} \
#             training.{lr=0.0003,epochs=20,batch_size=16}

# # recombination to range

# python -m experiments.composition with \
#             dataset.{twoshapes,modifiers='["noverlap"]'} \
#             dataset.{condition=recomb2range,variant=huecomb} \
#             model.{kim,sbd,compositional_op='interp'} \
#             training.{lr=0.0003,epochs=20,batch_size=16}

# python -m experiments.composition with \
#             dataset.{twoshapes,modifiers='["noverlap"]'} \
#             dataset.{condition=recomb2range,variant=huecomb} \
#             model.{kim,sbd,compositional_op='fixint'} \
            training.{lr=0.0003,epochs=20,batch_size=16}
