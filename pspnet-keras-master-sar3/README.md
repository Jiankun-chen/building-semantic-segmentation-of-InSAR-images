## PSPnet using for building semantic segmentation of InSAR images
---
**This code is adapted from the original PSPnet code, used as InSAR building semantic segmentation.
Address link of Source code：[https://github.com/bubbliiiing/pspnet-keras](https://github.com/bubbliiiing/pspnet-keras "PSPNet"). Each simulated InSAR sample contains three channels: master SAR image, slave SAR image, and interferometric phase image. They were fused into 3-channels samples.**
### Environment
tensorflow-gpu==1.14    

### Data set
**/VOCdevkit/VOC/2007/JPGEImages：** simulated InSAR building images for train. 

**/VOCdevkit/VOC/2007/SegmentataionClass：** GT labels--onehot coding. 

**/img：** both simulated InSAR building images and real airborne InSAR images for test.

### Segmentation results
**/results：** segmentation results on both the simulated InSAR building images test set and the real airborne InSAR images test set.

### logs
**/Output/model：** epXXX-lossXXX-val_lossXXX.h5 

