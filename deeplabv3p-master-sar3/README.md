# Deeplabv3+ using for building semantic segmentation of InSAR images

**This code is adapted from the original deeplabv3+ code, used as InSAR building semantic segmentation.
Address link of Source code：[https://github.com/rishizek/tensorflow-deeplab-v3-plus](https://github.com/rishizek/tensorflow-deeplab-v3-plus "deeplabv3+"). Each simulated InSAR sample contains three channels: master SAR image, slave SAR image, and interferometric phase image. They were fused into 3-channels samples. We can download and save the segmentation results from tensorboard.**

### Data set
**/dataset/VOCdevkit/VOC/2007/JPGEImages：** simulated InSAR building images. 

**/dataset/VOCdevkit/VOC/2007/SegmentataionClass：** GT labels--onehot coding. 

**/dataset/train.txt：** ID of training image.

**/dataset/val.txt：** ID of real airborne InSAR images for test.

**/dataset/val1.txt：** ID of simulated InSAR building images for test.

### Segmentation results
**/result/pred：** segmentation results on the simulated InSAR building images test set.

**/result1/pred：** segmentation results on the real airborne InSAR images test set.

### model
**/model：** model.ckpt-XXX 