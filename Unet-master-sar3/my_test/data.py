from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import scipy.misc
#from libtiff import TIFF


class myAugmentation(object):

    '''
    一个用于图像增强的类：
    首先：分别读取训练的图片和标签，然后将图片和标签合并用于下一个阶段使用
    然后：使用Keras的预处理来增强图像
    最后：将增强后的图片分解开，分为训练图片和训练标签
    '''

    def __init__(self, train_path="train", label_path="label", merge_path="merge", aug_merge_path="aug_merge", aug_train_path="aug_train", aug_label_path="aug_label", img_type="tif"):
        """
        使用glob从路径中得到所有的“.img_type”文件，初始化类：__init__()
        """
        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

    def Augmentation(self):
        """
        Start augmentation.....
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)
            img = x_t
            img = img.reshape((1,) + img.shape)
            savedir = path_aug_merge + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i))

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):
        # 增强一张图片的方法
        """
        augmentate one image
        """
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):
        # 将合在一起的图片分开
        """
        split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path

        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            train_imgs = glob.glob(path + "/*." + self.img_type)
            savedir = path_train + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + str(i)

            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]
                img = cv2.imread(imgname)
                img_train = img[:, :, 2]  # cv2 read image rgb->bgr
                img_label = img[:, :, 0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train)
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label" + "." + self.img_type, img_label)

    def splitTransform(self):
        # 拆分透视变换后的图像
        """
        split perspective transform images
        """
        #path_merge = "transform"
        #path_train = "transform/data/"
        #path_label = "transform/label/"

        path_merge = "deform/deform_norm2"
        path_train = "deform/train/"
        path_label = "deform/label/"

        train_imgs = glob.glob(path_merge + "/*." + self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:, :, 2]  # cv2 read image rgb->bgr
            img_label = img[:, :, 0]
            cv2.imwrite(path_train + midname + "." + self.img_type, img_train)
            cv2.imwrite(path_label + midname + "." + self.img_type, img_label)


class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path1 = "../deform/train/ang", data_path2="../deform/train/master", data_path3="../deform/train/slave", label_path="../deform/label", test_path1="../test/ang", test_path2="../test/master", test_path3="../test/slave", npy_path="../npydata", img_type="png"):
        # 数据处理类，初始化
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.data_path3 = data_path3
        self.label_path = label_path
        self.img_type = img_type
        self.test_path1 = test_path1
        self.test_path2 = test_path2
        self.test_path3 = test_path3
        self.npy_path = npy_path

# 创建训练数据
    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path1 + '/*.png')
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]
            imgs_ang = load_img(self.data_path1 + "/" + midname, grayscale=True)
            imgs_master = load_img(self.data_path2 + "/" + midname, grayscale=True)
            imgs_slave = load_img(self.data_path3 + "/" + midname, grayscale=True)
            label = load_img(self.label_path + "/" + midname, grayscale=True)
            imgs_ang = img_to_array(imgs_ang)
            imgs_master = img_to_array(imgs_master)
            imgs_slave = img_to_array(imgs_slave)
            imgs_c3 = np.concatenate((imgs_ang, imgs_master, imgs_slave), axis=2)
            label = img_to_array(label)

            imgdatas[i] = imgs_c3
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

# 创建测试数据
    def create_test_data(self):
        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        imgs = glob.glob(self.test_path1 + '/*.png')
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]
            imgs_ang = load_img(self.test_path1 + "/" + midname, grayscale=True)
            imgs_master = load_img(self.test_path2 + "/" + midname, grayscale=True)
            imgs_slave = load_img(self.test_path3 + "/" + midname, grayscale=True)
            imgs_ang = img_to_array(imgs_ang)
            imgs_master = img_to_array(imgs_master)
            imgs_slave = img_to_array(imgs_slave)
            imgs_c3 = np.concatenate((imgs_ang,imgs_master, imgs_slave), axis=2)

            imgdatas[i] = imgs_c3
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

# 加载训练图片与mask
    def load_train_data(self):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 180
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
    # 做一个阈值处理，输出的概率值大于0.5的就认为是对象，否则认为是背景
        #imgs_mask_train[imgs_mask_train > 0.5] = 1
        #imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

# 加载测试图片
    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 180
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test


if __name__ == "__main__":
    # 以下注释掉的部分为数据增强代码，通过他们可以将数据进行增强

    #aug = myAugmentation()
    # aug.Augmentation()
    # aug.splitMerge()
    # aug.splitTransform()

    mydata = dataProcess(256,256)
    mydata.create_train_data()
    mydata.create_test_data()

    imgs_train, imgs_mask_train = mydata.load_train_data()
    print(imgs_train.shape, imgs_mask_train.shape)
