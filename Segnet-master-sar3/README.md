# Segnet using for building semantic segmentation of InSAR images

**This code is adapted from the original Segnet code, used as InSAR building semantic segmentation.
Address link of Source code：[https://github.com/chankeh/CardiacSeg-master](https://github.com/chankeh/CardiacSeg-master "Segnet"). Each simulated InSAR sample contains three channels: master SAR image, slave SAR image, and interferometric phase image. We can download and save the segmentation results from tensorboard.**

### Data set
**/Data/Training：** simulated InSAR building images for train. 

**/Data/Test：** simulated InSAR building images for test.

**/Data/Test1：** real airborne InSAR images for test.

### Segmentation results
**/results：** segmentation results on the simulated InSAR building images test set (The rgb display results are in ../rgb sub folder).

**/results1：** segmentation results on the real airborne InSAR images test set (The rgb display results are in ../rgb sub folder).

### Model
**/Output/model：** model.ckpt-60310