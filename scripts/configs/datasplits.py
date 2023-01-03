"""
Data-splitting functions for each dataset.

These are the functions that exlcude combiantions from the datasets
in order to test different generalisation setttings. The splits are
organized in classes so they create different namespaces.

The general mechanism works by passing a condition and variant
parameter to the appropriate class. The splits are returned as index
values. That way the images and targets (when predicting a factor)
can be split in one call.

Each dataset contains a description of the generative factor names
and their values for quick referencing when adding more splits.
"""


import numpy as np
from functools import partial


def compose(mask, mod):
    def composed(factor_values, factor_classes):
        return (mask(factor_values, factor_classes) &
                mod(factor_values, factor_classes))
    return composed


class DataSplit:
    interp         = {}

    recomb2element = {}

    recomb2range   = {}

    extrp          = {}

    modifiers      = {}

    @classmethod
    def get_splits(cls, condition, variant, modifiers=None):
        try:
            if condition is None:
                masks = None, None
            elif condition == 'interp':
                masks = cls.interp[variant]()
            elif condition == 'recomb2element':
                masks = cls.recomb2element[variant]()
            elif condition == 'recomb2range':
                masks = cls.recomb2range[variant]()
            elif condition == 'extrp':
                masks = cls.extrp[variant]()
            else:
                raise ValueError('Unrecognized condition {}'.format(condition))
        except KeyError:
            raise ValueError('Unrecognized variant {} for condition {}'.format(
                variant, condition))

        if modifiers is not None:
            for mod in modifiers:
                if mod not in cls.modifiers:
                    raise ValueError('Unrecognized modifier {}'.format(mod))

                # If no mask, then modifier is only mask and
                # it is applied during training.
                if masks[0] is None:
                    masks = cls.modifiers[mod], None
                else:
                    modf = partial(compose, mod=cls.modifiers[mod])
                    masks = [(None if m is None else modf(m)) for m in masks]

        return masks


class DummySplits:
    @staticmethod
    def get_splits(condition=None, variant=None, modifiers=None):
        return None, None


## Shapes3D

class _Shapes3D:
    fh, wh, oh, scl, shp, orient = 0, 1, 2, 3, 4, 5

    # Modifies
    @classmethod
    def exclude_odd_ohues(cls, factor_values, factor_classes):
        return (factor_classes[:, cls.oh] % 2) == 0

    @classmethod
    def exclude_half_ohues(cls, factor_values, factor_classes):
        return factor_classes[:, cls.oh] < 5

    @classmethod
    def exclude_odd_wnf_hues(cls, factor_values, factor_classes):
        return (((factor_classes[:, cls.wh] % 2) == 0) &
                ((factor_classes[:, cls.fh] % 2) == 0))

    # Interpolation variants
    @classmethod
    def odd_ohue(cls):
        def train_mask(factor_values, factor_classes):
            return factor_classes[:, cls.oh] % 2 == 0

        def test_mask(factor_values, factor_classes):
            return factor_classes[:, cls.oh] % 2 == 1

        return train_mask, test_mask

    @classmethod
    def odd_wnf_hue(cls):
        def train_mask(factor_values, factor_classes):
            return cls.exclude_odd_wnf_hues(factor_values, factor_classes)

        def test_mask(factor_values, factor_classes):
            return ~cls.exclude_odd_wnf_hues(factor_values, factor_classes)

        return train_mask, test_mask

    # Extrapolation variants
    @classmethod
    def missing_fh_50(cls):
        def train_mask(factor_values, factor_classes):
            return factor_values[:, cls.fh] < 0.5

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask

    # Recombination to range
    @classmethod
    def ohue_to_whue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oh] >= 0.75) &
                    (factor_values[:, cls.wh] <= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def fhue_to_whue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.fh] >= 0.75) &
                    (factor_values[:, cls.wh] <= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_floor(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.fh] >= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_objh(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] >= 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def flanked_shape2ohue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] > 0.25) &
                    (factor_values[:, cls.oh] < 0.75))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_objh_quarter(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 3.0) &
                    (factor_values[:, cls.oh] <= 0.25))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_orientation(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:,cls.shp] == 3.0) &
                    (factor_values[:,cls.orient] >= 0))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    # Recombination to element
    @classmethod
    def leave1out(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oh] >= 0.8) &
                    (factor_values[:, cls.wh] >= 0.8) &
                    (factor_values[:, cls.fh] >= 0.8) &
                    (factor_values[:, cls.scl] >= 1.1) &
                    (factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.orient] > 20))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_ohue(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:, cls.shp] == 3.0) &
                    (factor_classes[:, cls.oh] == 2))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


class Shapes3D(DataSplit):
    """
    Boolean masks used to partition the Shapes3D dataset
    for each generalisation condition

    #=============================================================
    # Latent Dimension, Latent values
    #=============================================================
    # floor hue:        10 values linearly spaced in [0, 1)
    # wall hue:         10 values linearly spaced in [0, 1)
    # object hue:       10 values linearly spaced in [0, 1)
    # scale:            8 values linearly spaced in [0.75, 1.25]
    # shape:            4 values in [0, 1, 2, 3]
    # orientation:      15 values linearly spaced in [-30, 30]

    """
    interp         = {'odd_ohue'     : _Shapes3D.odd_ohue,
                      'odd_wnf_hue'  : _Shapes3D.odd_wnf_hue}

    recomb2element = {'shape2ohue'   : _Shapes3D.shape_ohue,
                      'leave1out'    : _Shapes3D.leave1out}

    recomb2range   = {'ohue2whue'      : _Shapes3D.ohue_to_whue,
                      'fhue2whue'      : _Shapes3D.fhue_to_whue,
                      'shape2ohue'     : _Shapes3D.shape_to_objh,
                      'shape2ohueq'    : _Shapes3D.shape_to_objh_quarter,
                      'shape2fhue'     : _Shapes3D.shape_to_floor,
                      'shape2orient'   : _Shapes3D.shape_to_orientation,
                      'shape2ohue_flnk': _Shapes3D.flanked_shape2ohue}

    extrp          = {'missing_fh'   : _Shapes3D.missing_fh_50}

    modifiers      = {'even_ohues'   : _Shapes3D.exclude_odd_ohues,
                      'half_ohues'   : _Shapes3D.exclude_half_ohues,
                      'even_wnf_hues': _Shapes3D.exclude_odd_wnf_hues}


## DSprites

class _DSprites:
    shp, scl, rot, tx, ty = 0, 1, 2, 3, 4
    a90, a120, a180, a240 = np.pi / 2, 4 * np.pi / 3, np.pi, 2 * np.pi / 3

    @classmethod
    def sparse_posX(cls, factor_values, factor_classes):
        return np.isin(factor_classes[:, cls.tx], [0, 7, 15, 23, 31])

    @classmethod
    def sparse_posY(cls, factor_values, factor_classes):
        return np.isin(factor_classes[:, cls.ty], [0, 7, 15, 23, 31])

    # masks for blank right side condition
    @classmethod
    def blank_side(cls):
        def blank_side_train(factor_values, factor_classes):
            return (factor_values[:, cls.tx] < 0.5)

        def blank_side_extrp(factor_values, factor_classes):
            return (factor_values[:, cls.tx] > 0.5)

        return blank_side_train, blank_side_extrp

    # Leave one shape out along translation dimension
    @classmethod
    def square_to_posX(cls):
        def shape2tx_train(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 1) |
                    (factor_values[:, cls.tx] < 0.5))

        def shape2tx_extrp(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.tx] > 0.5))

        return shape2tx_train, shape2tx_extrp

    @classmethod
    def ellipse_to_posX(cls):
        def shape2tx_train(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 2) |
                    (factor_values[:, cls.tx] < 0.5))

        def shape2tx_extrp(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 2) &
                    (factor_values[:, cls.tx] > 0.5))

        return shape2tx_train, shape2tx_extrp

    # leave1out_comb
    @classmethod
    def leave1out(cls):
        def leave1out_comb_test(factor_values, factor_classes):
            return ((factor_classes[:, cls.shp] == 1) &
                    (factor_values[:, cls.scl] > 0.6) &
                    (factor_values[:, cls.rot] > 0.0) &
                    (factor_values[:, cls.rot] < cls.a120) &
                    (factor_values[:, cls.rot] > cls.a240) &
                    (factor_values[:, cls.tx] > 0.66) &
                    (factor_values[:, cls.ty] > 0.66))

        def leave1out_comb_train(factor_values, factor_classes):
            return ~leave1out_comb_test(factor_values, factor_classes)

        return leave1out_comb_train, leave1out_comb_test

    @classmethod
    def square_to_scale(cls):
        def shape2tx_train(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 1) |
                    (factor_values[:, cls.scl] < 0.75))

        def shape2tx_extrp(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.scl] > 0.75))

        return shape2tx_train, shape2tx_extrp

    @classmethod
    def centered_sqr2tx(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.tx] > 0.25) &
                    (factor_values[:, cls.tx] < 0.75))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def rshift_sqr2tx(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] == 1) &
                    (factor_values[:, cls.tx] > 0.40) &
                    (factor_values[:, cls.tx] < 0.90))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def flanked_sqr2tx(cls):
        def train_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 1) |
                    (factor_values[:, cls.tx] < 0.25) |
                    (factor_values[:, cls.tx] > 0.75))

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def lshift_sqrt2tx(cls):
        def train_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] != 2) |
                    ((factor_values[:, cls.tx] > 0.10) &
                     (factor_values[:, cls.tx] < 0.60)))

        def test_mask(factor_values, factor_classes):
            return ~train_mask(factor_values, factor_classes)

        return train_mask, test_mask


class DSprites(DataSplit):
    """
    Boolean masks used to partition the dSprites dataset
    for each generalisation condition

    #=============================================================
    # Latent Dimension, Latent values
    #=============================================================
    # Luminence       - 255
    # Scale           - [0.5, 1] split into 6 values
    # Angle           - [0  , 2pi] split into 40 values
    # Translation X   - [0  , 1] split into 32 values
    # Translation Y   - [0  , 1] split into 32 values
    """

    interp         = { }

    recomb2element = {'leave1out'  : _DSprites.leave1out}

    recomb2range   = {'sqr2tx'     : _DSprites.square_to_posX,
                      'ell2tx'     : _DSprites.ellipse_to_posX,
                      'sqr2scl'    : _DSprites.square_to_scale,
                      'sqr2tx_cent': _DSprites.centered_sqr2tx,
                      'sqr2tx_flnk': _DSprites.flanked_sqr2tx,
                      'sqr2tx_rs'  : _DSprites.rshift_sqr2tx,
                      'sqr2tx_ls'  : _DSprites.lshift_sqrt2tx}

    extrp          = {'blank_side' : _DSprites.blank_side}

    modifiers      = {'sparse_posX': _DSprites.sparse_posX}

##################################### MPI3D ###################################


class _MPID3D:
    oc, shp, sz, camh, bkg, hx, vx = 0, 1, 2, 3, 4, 5, 6

    # Modifiers
    @classmethod
    def remove_redundant_shapes(cls, factor_values, factor_classes):
        return ~np.isin(factor_values[:, cls.shp], np.array([0,3]))

    @classmethod
    def fix_hx(cls, factor_values, factor_classes):
        return factor_values[:, cls.hx] == 0

    @classmethod
    def lhalf_hx(cls, factor_values, factor_classes):
        return factor_values[:, cls.hx] < 20

    @classmethod
    def even_vx(cls, factor_values, factor_classes):
        return factor_classes[:, cls.vx] % 2 == 1

    # Extrapolation
    @classmethod
    def exclude_horz_gt20(cls):
        def train_mask(factor_values, factor_classes):
            return factor_values[:, cls.hx] < 20

        def test_mask(factor_values, factor_classes):
            return (factor_values[:, cls.hx] > 20)

        return train_mask, test_mask

    @classmethod
    def exclude_objc_gt3(cls):
        def train_mask(factor_values, factor_classes):
            return factor_values[:, cls.oc] <= 3

        def test_mask(factor_values, factor_classes):
            return factor_values[:, cls.oc] > 3

        return train_mask, test_mask

    # Recombination to element
    @classmethod
    def cylinder_to_horizontal_axis(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.hx] < 20) &
                    (factor_values[:, cls.shp] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @classmethod
    def cylinder_to_vertial_axis(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.vx] < 20) &
                    (factor_values[:, cls.shp] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @classmethod
    def cylinder_to_background(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.bkg] > 0) &
                    (factor_values[:, cls.shp] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @classmethod
    def redobject2hz(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.hx] < 20) &
                    (factor_values[:, cls.oc] == 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    @classmethod
    def background_to_cylinder(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.shp] >= 4) &
                    (factor_values[:, cls.bkg] == 2))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def background2obj_color(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.bkg] == 2) &
                    (factor_values[:, cls.oc] > 2))

        def training_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return training_mask, test_mask

    # Recombination to element
    @classmethod
    def leave1out(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.oc] == 5) &
                    (factor_values[:, cls.shp] == 2) &
                    (factor_values[:, cls.sz] == 1) &
                    (factor_values[:, cls.camh] == 1) &
                    (factor_values[:, cls.bkg] == 1) &
                    (factor_values[:, cls.hx] > 35) &
                    (factor_values[:, cls.vx] > 35))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


class MPI3D(DataSplit):
    """
    # Boolean masks used to partition the MPI datasets
    # for each generalisation condition

    #==========================================================================
    # Latent Dimension,    Latent values                               N vals
    #==========================================================================
    # object color:        white=0, green=1, red=2,                     6
    #                      blue=3, brown=4, olive=5
    # object shape:        cone=0, cube=1, cylinder=2,                  6
    #                      hexagonal=3, pyramid=4, sphere=5
    # object size:         small=0, large=1                             2
    # camera height:       top=0, center=1, bottom=2                    3
    # background color:    purple=0, sea green=1, salmon=2              3
    # horizontal axis:     linearly spaced in [0, 40)                  40
    # vertical axis:       linearly spaced in [0, 40)                  40
    """

    interp         = { }

    recomb2element = {'leave1out'  : _MPID3D.leave1out}

    recomb2range   = {'cyl2hx'     : _MPID3D.cylinder_to_horizontal_axis,
                      'cyl2vx'     : _MPID3D.cylinder_to_vertial_axis,
                      'redoc2hx'   : _MPID3D.redobject2hz,
                      'bkg2cyl'    : _MPID3D.background_to_cylinder}

    extrp          = {'horz_gt20'  : _MPID3D.exclude_horz_gt20,
                      'objc_gt3'   : _MPID3D.exclude_objc_gt3}

    modifiers      = {'four_shapes': _MPID3D.remove_redundant_shapes,
                      'fix_hx'     : _MPID3D.fix_hx,
                      'lhalf_hx'   : _MPID3D.lhalf_hx,
                      'even_vx'    : _MPID3D.even_vx}



############################## Spriteworld #############################

class _Circles:
    px, py = 0, 1

    @classmethod
    def midpos(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.px] >= 0.35) &
                    (factor_values[:, cls.px] <= 0.65) &
                    (factor_values[:, cls.py] >= 0.35) &
                    (factor_values[:, cls.py] <= 0.65))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_posx(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.px] > 0.5) &
                    (factor_values[:, cls.py] > 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


class Circles(DataSplit):
    recomb2range = {'midpos'  : _Circles.midpos,
                    'shape2px': _Circles.shape_to_posx}


class _Simple:
    px, py, shape = 0, 1, 2

    @classmethod
    def midpos(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_values[:, cls.px] >= 0.35) &
                    (factor_values[:, cls.px] <= 0.65) &
                    (factor_values[:, cls.py] >= 0.35) &
                    (factor_values[:, cls.py] <= 0.65) &
                    (factor_classes[:,cls.shape] == 1))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask

    @classmethod
    def shape_to_posx(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:, cls.shape] == 1) &
                    (factor_values[:, cls.px] > 0.5) &
                    (factor_values[:, cls.py] > 0.5))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


class Simple(DataSplit):
    recomb2range = {'shape2px': _Simple.shape_to_posx,
                    'midpos'  : _Simple.midpos}


class _TwoShapes:
    o1_px, o1_py, o1_shape, o1_hue = 0, 1, 2, 3
    o2_px, o2_py, o2_shape, o2_hue = 4, 5, 6, 7

    @classmethod
    def non_overlapping(cls, factor_values, factor_classes):
        dx = np.abs(factor_values[:,cls.o1_px] - factor_values[:,cls.o2_px])
        dy = np.abs(factor_values[:,cls.o1_py] - factor_values[:,cls.o2_py])

        return (dx > 0.2) & (dy > 0.2)

    @classmethod
    def unique_side(cls, factor_values, factor_classes):
        return ((factor_values[:,cls.o1_px] <= 0.35) &
                (factor_values[:,cls.o2_px] >= 0.65))

    @classmethod
    def hue_combs(cls):
        def test_mask(factor_values, factor_classes):
            return ((factor_classes[:, cls.o1_hue] < 2) &
                    (factor_classes[:, cls.o2_hue] > 2))

        def train_mask(factor_values, factor_classes):
            return ~test_mask(factor_values, factor_classes)

        return train_mask, test_mask


class TwoShapes(DataSplit):
    modifiers    = {'noverlap': _TwoShapes.non_overlapping}

    recomb2range = {'huecomb' : _TwoShapes.hue_combs}
