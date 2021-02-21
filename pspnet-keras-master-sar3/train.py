import os
import time
import keras
import numpy as np
from nets.pspnet import pspnet
from nets.pspnet_training import Generator, dice_loss_with_CE, CE
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.metrics import categorical_accuracy
from keras import backend as K
from PIL import Image
from utils.metrics import Iou_score, f_score
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mobilenet_freeze = 146
resnet_freeze = 172

if __name__ == "__main__":     
    inputs_size = [256,256,3]
    log_dir = "logs/"
    #---------------------#
    #   分类个数+1
    #   2+1
    #---------------------#
    num_classes = 3
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = True
    #---------------------#
    #   主干网络选择
    #   mobilenet
    #   resnet50
    #---------------------#
    backbone = "mobilenet"
    #---------------------#
    #   是否使用辅助分支
    #   会占用大量显存
    #---------------------#
    aux_branch = False
    #---------------------#
    #   下采样的倍数
    #   16显存占用小
    #   8显存占用大
    #---------------------#
    downsample_factor =8

    # 获取model
    model = pspnet(num_classes,inputs_size,downsample_factor=downsample_factor,backbone=backbone,aux_branch=aux_branch)
    model.summary()

    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    model_path = "./model_data/pspnet_mobilenetv2.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt","r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt","r") as f:
        val_lines = f.readlines()
        
    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1)
    # tensorboard
    tensorboard = TensorBoard(log_dir=log_dir)

    if backbone=="mobilenet":
        freeze_layers = mobilenet_freeze
    else:
        freeze_layers = resnet_freeze

    for i in range(freeze_layers): 
        model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        Batch_size = 8
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate(False)
        # 开始训练
        model.fit_generator(gen,
                steps_per_epoch=max(2, len(train_lines)//Batch_size)//4,
                validation_data=gen_val,
                validation_steps=max(4, len(val_lines)//Batch_size)//4,
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])
    
    for i in range(freeze_layers): 
        model.layers[i].trainable = True

    if True:
        lr = 1e-5
        Freeze_Epoch = 50
        Unfreeze_Epoch = 500
        Batch_size = 4
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate(False)
        # 开始训练
        model.fit_generator(gen,
                steps_per_epoch=max(2, len(train_lines)//Batch_size),
                validation_data=gen_val,
                validation_steps=max(4, len(val_lines)//Batch_size),
                epochs=Unfreeze_Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])

                
