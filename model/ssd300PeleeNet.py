# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:33:53 2019
PeleeNet
@author: ThinkPad
"""
from keras.layers import Conv2D,BatchNormalization,Activation,Concatenate,Dense,Flatten
from keras.layers import MaxPooling2D,Add,Input,ZeroPadding2D,AveragePooling2D,GlobalAveragePooling2D
from keras.layers import Reshape,concatenate
from keras.models import Model
from keras import backend as K
from ssd_layers import PriorBox

def Conv_bn_relu(inp, oup, kernel_size=3, stride=1, pad=1,use_relu = True):
    if pad !=0 :
        x=ZeroPadding2D(padding=(pad,pad))(inp)
    else:
        x=inp
    
    x=Conv2D(oup,kernel_size=(kernel_size,kernel_size),strides=(stride,stride),
                 use_bias=False,padding='valid')(x)
    x=BatchNormalization()(x)
    if use_relu:
        x=Activation('relu')(x)
    return x



def res_block(inp,planes=128):
    stem1=Conv_bn_relu(inp,planes,1,1,0)
    stem1=Conv_bn_relu(stem1,planes,3,1,1)
    stem1=Conv_bn_relu(stem1,planes*2,1,1,0,False)
    
    stem2=Conv_bn_relu(inp,planes*2,1,1,0,False)
    out=Add()([stem1,stem2])
    return Activation('relu')(out)

def StemBlock(inp,num_init_features=32):
    stem1=Conv_bn_relu(inp, oup=num_init_features, kernel_size=3, stride=2, pad=1,use_relu = True)
    stem2=Conv_bn_relu(stem1, oup=int(num_init_features/2), kernel_size=1, stride=1, pad=0,use_relu = True)
    stem2=Conv_bn_relu(stem2, oup=num_init_features, kernel_size=3, stride=2, pad=1,use_relu = True)
    stem3=MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(stem1)
    out=Concatenate(axis=-1)([stem2,stem3])
    out=Conv_bn_relu(out, oup=num_init_features, kernel_size=1, stride=1, pad=0,use_relu = True)
    return out
    

def DenseBlock(inp,inter_channel,growth_rate):
    cb1_a=Conv_bn_relu(inp,inter_channel,1,1,0)
    cb1_b=Conv_bn_relu(cb1_a,growth_rate,3,1,1)
    
    cb2_a=Conv_bn_relu(inp,inter_channel,1,1,0)
    cb2_b=Conv_bn_relu(cb2_a,growth_rate,3,1,1)
    cb2_c=Conv_bn_relu(cb2_b,growth_rate,3,1,1)
    return Concatenate(axis=-1)([inp,cb1_b,cb2_c])

def TransitionBlock(inp, oup,with_pooling= True):
    x=Conv_bn_relu(inp,oup,1,1,0)
    if with_pooling:
        x=AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
    return x


def _make_dense_transition(inputs,half_growth_rate,total_filter, inter_channel, ndenseblocks,with_pooling= True):
    x=inputs
    for i in range(ndenseblocks):
        x=DenseBlock(x,inter_channel,half_growth_rate)
    x=TransitionBlock(x,total_filter,with_pooling)
    return x


def prediction(x,i,num_priors,min_s,max_s,aspect,num_classes,img_size):
    a=Conv2D(num_priors*4,(1,1),padding='valid',name=str(i)+'_mbox_loc')(x)
    mbox_loc_flat=Flatten(name=str(i)+'_mbox_loc_flat')(a)
    b=Conv2D(num_priors*num_classes,(1,1),padding='valid',name=str(i)+'_mbox_conf')(x)
    mbox_conf_flat=Flatten(name=str(i)+'_mbox_conf_flat')(b)
    mbox_priorbox=PriorBox(img_size,min_size=min_s,max_size=max_s,aspect_ratios=aspect,variances=[0.1,0.1,0.2,0.2],name=str(i)+'_mbox_priorbox')(x)
    return mbox_loc_flat,mbox_conf_flat,mbox_priorbox



def SSD(input_shape,num_classes=21, num_init_features=32,growthRate=32,
             nDenseBlocks = [3,4,8,6], bottleneck_width=[1,2,4,4]):
    inter_channel =list()
    total_filter =list()
    half_growth_rate = int(growthRate / 2)
    
    inp=Input(shape=input_shape)
    img_size=(input_shape[1],input_shape[0])
    stage=StemBlock(inp,num_init_features)
    #print(stage.shape)
    #pwconv3=stage
    for i,b_w in enumerate(bottleneck_width):
        inter_channel.append(int(half_growth_rate*b_w/4)*4)
        if i==0:
            total_filter.append(num_init_features + growthRate * nDenseBlocks[i])           
        else:
            total_filter.append(total_filter[i-1] + growthRate * nDenseBlocks[i])

        if i == len(nDenseBlocks)-1:
            with_pooling = False
        else:
            with_pooling = True
        stage=_make_dense_transition(stage,half_growth_rate,total_filter[i],inter_channel[i],nDenseBlocks[i],with_pooling=with_pooling)
        #print(stage.shape)
        if i == 1:
            pwconv3=stage
    pwconv4=stage
    #5*5
    pwconv5=Conv_bn_relu(stage, 256, kernel_size=1, stride=1, pad=0,use_relu = True)
    pwconv5=Conv_bn_relu(pwconv5, 256, kernel_size=3, stride=2, pad=1,use_relu = True)
    #3*3
    pwconv6=Conv_bn_relu(pwconv5, 128, kernel_size=1, stride=1, pad=0,use_relu = True)
    pwconv6=Conv_bn_relu(pwconv6, 256, kernel_size=3, stride=1, pad=0,use_relu = True)
    #1*1
    pwconv7=Conv_bn_relu(pwconv6, 128, kernel_size=1, stride=1, pad=0,use_relu = True)
    pwconv7=Conv_bn_relu(pwconv7, 256, kernel_size=3, stride=1, pad=0,use_relu = True)
    
    pwconv3=res_block(pwconv3,planes=256)
    pwconv4=res_block(pwconv4,planes=256)
    pwconv5=res_block(pwconv5,planes=256)
    pwconv6=res_block(pwconv6,planes=256)
    pwconv7=res_block(pwconv7,planes=256)
    
    pwconv3_mbox_loc_flat, pwconv3_mbox_conf_flat, pwconv3_mbox_priorbox = prediction(pwconv3, 3, 3, 30.4 ,None ,[2],num_classes, img_size)
    pwconv4_mbox_loc_flat, pwconv4_mbox_conf_flat, pwconv4_mbox_priorbox = prediction(pwconv4, 4, 6, 60.8,112.5,[2, 3], num_classes, img_size)
    pwconv5_mbox_loc_flat, pwconv5_mbox_conf_flat, pwconv5_mbox_priorbox = prediction(pwconv4, 5, 6, 112.5,164.2,[2, 3], num_classes, img_size)
    pwconv6_mbox_loc_flat, pwconv6_mbox_conf_flat, pwconv6_mbox_priorbox = prediction(pwconv5, 6, 6, 164.2,215.8,[2, 3], num_classes, img_size)
    pwconv7_mbox_loc_flat, pwconv7_mbox_conf_flat, pwconv7_mbox_priorbox = prediction(pwconv6, 7, 6, 215.8,267.4,[2, 3], num_classes, img_size)
    pwconv8_mbox_loc_flat, pwconv8_mbox_conf_flat, pwconv8_mbox_priorbox = prediction(pwconv7, 8, 6, 267.4,300.0,[2, 3],num_classes, img_size)


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
    
    model = Model(inputs=inp,outputs=predictions)
    
    print(predictions.shape)
    #print(pwconv3.shape)
    #print(pwconv4.shape)
    #print(pwconv5.shape)
    #print(pwconv6.shape)
    #print(pwconv7.shape)
    return model
            
print(SSD((300,300,3)).summary()) 














