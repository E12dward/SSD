# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:58:24 2019
shufflenet V2
@author: ThinkPad
"""

import numpy as np
from keras.utils import plot_model
from keras.engine.topology import get_source_inputs
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D,Concatenate
from keras.layers import Activation, Dense
from keras.layers import concatenate,Reshape,Flatten
from keras.models import Model
import keras.backend as K
from ssd_layers import PriorBox


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x



def relu6(x):
    return K.relu(x, max_value=6)

def LiteConv(x,i,filter_num):
    x = Conv2D(filter_num//2, (1, 1), padding='same', use_bias=False, name=str(i)+'_pwconv1')(x)
    x = BatchNormalization(momentum=0.99,name=str(i)+'_pwconv1_bn')(x)
    x = Activation('relu', name=str(i)+'_pwconv1_act')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name=str(i)+'_dwconv2')(x)
    x = BatchNormalization(momentum=0.99,name=str(i) + '_sepconv2_bn')(x)
    x = Activation('relu', name=str(i) + '_sepconv2_act')(x)
    net = Conv2D(filter_num, (1, 1), padding='same', use_bias=False, name=str(i) + '_pwconv3')(x)
    x = BatchNormalization(momentum=0.99,name=str(i) + '_pwconv3_bn')(net)
    x = Activation('relu', name=str(i) + '_pwconv3_act')(x)
    #print(x.shape)
    return x,net

def Conv(x,filter_num):
    net = Conv2D(filter_num,kernel_size=1,strides=(1,1),use_bias=False,name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(net)
    x = Activation(relu6, name='Conv_1_relu')(x)
    #print(x.shape)
    return x,net


def prediction(x,i,num_priors,min_s,max_s,aspect,num_classes,img_size):
    a=Conv2D(num_priors*4,(3,3),padding='same',name=str(i)+'_mbox_loc')(x)
    mbox_loc_flat=Flatten(name=str(i)+'_mbox_loc_flat')(a)
    b=Conv2D(num_priors*num_classes,(3,3),padding='same',name=str(i)+'_mbox_conf')(x)
    mbox_conf_flat=Flatten(name=str(i)+'_mbox_conf_flat')(b)
    mbox_priorbox=PriorBox(img_size,min_size=min_s,max_size=max_s,aspect_ratios=aspect,variances=[0.1,0.1,0.2,0.2],name=str(i)+'_mbox_priorbox')(x)
    return mbox_loc_flat,mbox_conf_flat,mbox_priorbox



def SSD(input_shape,num_classes,scale_factor=1.0,
        num_shuffle_units=[3,7,3],bottleneck_ratio=1):
    
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)
    
    img_size=(input_shape[1],input_shape[0])
    Input0=Input(input_shape)

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(Input0)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)
        if stage == 1:
            pwconv3=x

    x, pwconv4 = Conv(x, 1280)
    x, pwconv5 = LiteConv(x, 5, 512)
    x, pwconv6 = LiteConv(x, 6, 256)
    x, pwconv7 = LiteConv(x, 7, 128)
    x, pwconv8 = LiteConv(x, 8, 128)
    '''
    model = Model(inputs=Input0,  outputs=pwconv8)
    print(pwconv3.shape)
    print(pwconv4.shape)
    print(pwconv5.shape)
    print(pwconv6.shape)
    print(pwconv7.shape)
    print(pwconv8.shape)
    '''
    pwconv3_mbox_loc_flat, pwconv3_mbox_conf_flat, pwconv3_mbox_priorbox = prediction(pwconv3, 3, 3, 60.0 ,None ,[2],num_classes, img_size)
    pwconv4_mbox_loc_flat, pwconv4_mbox_conf_flat, pwconv4_mbox_priorbox = prediction(pwconv4, 4, 6, 105.0,150.0,[2, 3], num_classes, img_size)
    pwconv5_mbox_loc_flat, pwconv5_mbox_conf_flat, pwconv5_mbox_priorbox = prediction(pwconv5, 5, 6, 150.0,195.0,[2, 3], num_classes, img_size)
    pwconv6_mbox_loc_flat, pwconv6_mbox_conf_flat, pwconv6_mbox_priorbox = prediction(pwconv6, 6, 6, 195.0,240.0,[2, 3], num_classes, img_size)
    pwconv7_mbox_loc_flat, pwconv7_mbox_conf_flat, pwconv7_mbox_priorbox = prediction(pwconv7, 7, 6, 240.0,285.0,[2, 3], num_classes, img_size)
    pwconv8_mbox_loc_flat, pwconv8_mbox_conf_flat, pwconv8_mbox_priorbox = prediction(pwconv8, 8, 6, 285.0,300.0,[2, 3],num_classes, img_size)


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


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = SSD((300, 300, 3), 21)
    #plot_model(model, to_file='shufflenetv2.png', show_layer_names=True, show_shapes=True)
    print(model.summary())
    pass
