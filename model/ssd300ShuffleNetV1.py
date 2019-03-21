# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:52:55 2019
shufflenet V1
@author: ThinkPad
"""

from keras import backend as K
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D,GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D
import numpy as np
from keras.layers import Flatten,concatenate,Reshape
from ssd_layers import PriorBox



def _block(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
    """
    creates a bottleneck block containing `repeat + 1` shuffle units
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    channel_map: list
        list containing the number of output channels for a stage
    repeat: int(1)
        number of repetitions for a shuffle unit with stride 1
    groups: int(1)
        number of groups per channel
    bottleneck_ratio: float
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    stage: int(1)
        stage number
    Returns
    -------
    """
    x = _shuffle_unit(x, in_channels=channel_map[stage - 2],
                      out_channels=channel_map[stage - 1], strides=2,
                      groups=groups, bottleneck_ratio=bottleneck_ratio,
                      stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = _shuffle_unit(x, in_channels=channel_map[stage - 1],
                          out_channels=channel_map[stage - 1], strides=1,
                          groups=groups, bottleneck_ratio=bottleneck_ratio,
                          stage=stage, block=(i + 1))

    return x


def _shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
    """
    creates a shuffleunit
    Parameters
    ----------
    inputs:
        Input tensor of with `channels_last` data format
    in_channels:
        number of input channels
    out_channels:
        number of output channels
    strides:
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
    groups: int(1)
        number of groups per channel
    bottleneck_ratio: float
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    stage: int(1)
        stage number
    block: int(1)
        block number
    Returns
    -------
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    prefix = 'stage%d/block%d' % (stage, block)

    #if strides >= 2:
        #out_channels -= in_channels

    # default: 1/4 of the output channel of a ShuffleNet Unit
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage == 2 and block == 1 else groups)

    x = _group_conv(inputs, in_channels, out_channels=bottleneck_channels,
                    groups=(1 if stage == 2 and block == 1 else groups),
                    name='%s/1x1_gconv_1' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(x)
    x = Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

    x = Lambda(channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False,
                        strides=strides, name='%s/1x1_dwconv_1' % prefix)(x)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(x)

    x = _group_conv(x, bottleneck_channels, out_channels=out_channels if strides == 1 else out_channels - in_channels,
                    groups=groups, name='%s/1x1_gconv_2' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(x)

    if strides < 2:
        ret = Add(name='%s/add' % prefix)([x, inputs])
    else:
        avg = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
        ret = Concatenate(bn_axis, name='%s/concat' % prefix)([x, avg])

    ret = Activation('relu', name='%s/relu_out' % prefix)(ret)

    return ret




def _group_conv(x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
    """
    grouped convolution
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    in_channels:
        number of input channels
    out_channels:
        number of output channels
    groups:
        number of groups per channel
    kernel: int(1)
        An integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    stride: int(1)
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for all spatial dimensions.
    name: str
        A string to specifies the layer name
    Returns
    -------
    """
    if groups == 1:
        return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                      use_bias=False, strides=stride, name=name)(x)

    # number of intput channels per group
    ig = in_channels // groups
    group_list = []

    assert out_channels % groups == 0

    for i in range(groups):
        offset = i * ig
        group = Lambda(lambda z: z[:, :, :, offset: offset + ig], name='%s/g%d_slice' % (name, i))(x)
        group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
                                 use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
    return Concatenate(name='%s/concat' % name)(group_list)


def channel_shuffle(x, groups):
    """
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    groups: int
        number of groups per channel
    Returns
    -------
        channel shuffled output tensor
    Examples
    --------
    Example for a 1D Array with 3 groups
    >>> d = np.array([0,1,2,3,4,5,6,7,8])
    >>> x = np.reshape(d, (3,3))
    >>> x = np.transpose(x, [1,0])
    >>> x = np.reshape(x, (9,))
    '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])

    return x


def prediction(x,i,num_priors,min_s,max_s,aspect,num_classes,img_size):
    a=Conv2D(num_priors*4,(3,3),padding='same',name=str(i)+'_mbox_loc')(x)
    mbox_loc_flat=Flatten(name=str(i)+'_mbox_loc_flat')(a)
    b=Conv2D(num_priors*num_classes,(3,3),padding='same',name=str(i)+'_mbox_conf')(x)
    mbox_conf_flat=Flatten(name=str(i)+'_mbox_conf_flat')(b)
    mbox_priorbox=PriorBox(img_size,min_size=min_s,max_size=max_s,aspect_ratios=aspect,variances=[0.1,0.1,0.2,0.2],name=str(i)+'_mbox_priorbox')(x)
    return mbox_loc_flat,mbox_conf_flat,mbox_priorbox


def SSD(input_shape,num_classes,scale_factor=1.0, 
        groups=3,num_shuffle_units=[3, 7, 3],bottleneck_ratio=0.25, classes=21):
    img_size=(input_shape[1],input_shape[0])
    input_shape=(input_shape[1],input_shape[0],3)
    Input0 = Input(input_shape)
    
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support ')

    name = "ShuffleNet_%.2gX_g%d_br_%.2g_%s" % (scale_factor, groups, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))

    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    if groups not in out_dim_stage_two:
        raise ValueError("Invalid number of groups.")

    if not (float(scale_factor) * 4).is_integer():
        raise ValueError("Invalid value for scale_factor. Should be x over 4.")

    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)


    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same',
               use_bias=False, strides=(2, 2), activation="relu", name="conv1")(Input0)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="maxpool1")(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(0, len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = _block(x, out_channels_in_stage, repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   groups=groups, stage=stage + 2)
        if stage == 1:
            pwconv3=x
            
        if stage == 2:
            pwconv4=x
            
    conv18_1=Conv2D(256,kernel_size=(1,1),padding='valid',strides=(1,1))(x)
    conv18_1=BatchNormalization()(conv18_1)
    conv18_1=Activation('relu')(conv18_1)
    conv18_2=Conv2D(512,kernel_size=(3,3),padding='same',strides=(2,2))(conv18_1)
    conv18_2=BatchNormalization()(conv18_2)
    conv18_2=Activation('relu')(conv18_2)
    
    conv19_1=Conv2D(128,kernel_size=(1,1),padding='valid',strides=(1,1))(conv18_2)
    conv19_1=BatchNormalization()(conv19_1)
    conv19_1=Activation('relu')(conv19_1)
    conv19_2=Conv2D(256,kernel_size=(3,3),padding='same',strides=(2,2))(conv19_1)
    conv19_2=BatchNormalization()(conv19_2)
    conv19_2=Activation('relu')(conv19_2)
    
    conv20_1=Conv2D(128,kernel_size=(1,1),padding='valid',strides=(1,1))(conv19_2)
    conv20_1=BatchNormalization()(conv20_1)
    conv20_1=Activation('relu')(conv20_1)
    conv20_2=Conv2D(256,kernel_size=(3,3),padding='same',strides=(2,2))(conv20_1)
    conv20_2=BatchNormalization()(conv20_2)
    conv20_2=Activation('relu')(conv20_2)
    
    conv21_1=Conv2D(64,kernel_size=(1,1),padding='valid',strides=(1,1))(conv20_2)
    conv21_1=BatchNormalization()(conv21_1)
    conv21_1=Activation('relu')(conv21_1)
    conv21_2=Conv2D(128,kernel_size=(3,3),padding='same',strides=(2,2))(conv21_1)
    conv21_2=BatchNormalization()(conv21_2)
    conv21_2=Activation('relu')(conv21_2)
    
    print(pwconv3.shape)
    print(pwconv4.shape)
    print(conv18_2.shape)
    print(conv19_2.shape)
    print(conv20_2.shape)
    print(conv21_2.shape)
    
    pwconv3_mbox_loc_flat, pwconv3_mbox_conf_flat, pwconv3_mbox_priorbox = prediction(pwconv3, 3, 3, 60.0 ,None ,[2],num_classes, img_size)
    pwconv4_mbox_loc_flat, pwconv4_mbox_conf_flat, pwconv4_mbox_priorbox = prediction(pwconv4, 4, 6, 105.0,150.0,[2, 3], num_classes, img_size)
    pwconv5_mbox_loc_flat, pwconv5_mbox_conf_flat, pwconv5_mbox_priorbox = prediction(conv18_2, 5, 6, 150.0,195.0,[2, 3], num_classes, img_size)
    pwconv6_mbox_loc_flat, pwconv6_mbox_conf_flat, pwconv6_mbox_priorbox = prediction(conv19_2, 6, 6, 195.0,240.0,[2, 3], num_classes, img_size)
    pwconv7_mbox_loc_flat, pwconv7_mbox_conf_flat, pwconv7_mbox_priorbox = prediction(conv20_2, 7, 6, 240.0,285.0,[2, 3], num_classes, img_size)
    pwconv8_mbox_loc_flat, pwconv8_mbox_conf_flat, pwconv8_mbox_priorbox = prediction(conv21_2, 8, 6, 285.0,300.0,[2, 3],num_classes, img_size)


    # Gather all predictions
    mbox_loc = concatenate(
        [pwconv3_mbox_loc_flat, pwconv4_mbox_loc_flat, pwconv5_mbox_loc_flat, pwconv6_mbox_loc_flat,
         pwconv7_mbox_loc_flat, pwconv8_mbox_loc_flat], axis=1, name='mbox_loc')
    mbox_conf = concatenate(
        [pwconv3_mbox_conf_flat, pwconv4_mbox_conf_flat, pwconv5_mbox_conf_flat, pwconv6_mbox_conf_flat,
         pwconv7_mbox_conf_flat, pwconv8_mbox_conf_flat], axis=1, name='mbox_conf')
    mbox_priorbox = concatenate(
        [pwconv3_mbox_priorbox, pwconv4_mbox_priorbox, pwconv5_mbox_priorbox, pwconv6_mbox_priorbox,
         pwconv7_mbox_priorbox, pwconv8_mbox_priorbox], axis=1, name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc,mbox_conf,mbox_priorbox],axis=2,name='predictions')
    
    model = Model(inputs=Input0,  outputs=predictions)
    return model
   





print(SSD((300,300,3),21).summary())