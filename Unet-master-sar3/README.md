# Unet using for building semantic segmentation of InSAR images

**This code is adapted from the original UNET code, used as InSAR building semantic segmentation.
Address link of Source code：[https://github.com/decouples/Unet](https://github.com/decouples/Unet "Source code link"). Each simulated InSAR sample contains three channels: master SAR image, slave SAR image, and interferometric phase image.**

### Data set
**/deform/train：** simulated InSAR building images for train. 

**/test1：** simulated InSAR building images for test.

**/test：** real airborne InSAR images for test.

### Segmentation results
**/results1：** segmentation results on the simulated InSAR building images test set (The rgb display results are in ../rgb sub folder).

**/results：** segmentation results on the real airborne InSAR images test set (The rgb display results are in ../rgb sub folder).

### Model
**/mytest：** my_unet.hdf5