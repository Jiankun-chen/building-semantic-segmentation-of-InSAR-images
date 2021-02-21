# RefineNet using for building semantic segmentation of InSAR images

**This code is adapted from the original RefineNet code, used as InSAR building semantic segmentation.
Address link of Source code：[https://github.com/markshih91/refinenet_pytorch](https://github.com/markshih91/refinenet_pytorch "refinenet"). Each simulated InSAR sample contains three channels: master SAR image, slave SAR image, and interferometric phase image.**

### Data set
**/data/nyu_depths：** not used. 

**/data/nyu_images_ang：** training simulated InSAR building interferometric phase images.

**/data/nyu_images_master：** training simulated InSAR building master images.

**/data/nyu_images_slave：** training simulated InSAR building slave images.

**/data/predict：** real airborne InSAR images for test.

**/data/predict1：** simulated InSAR building images for test.

### Segmentation results
**/data/predict/labels：** segmentation results on the real airborne InSAR images test set (The rgb display results are in ../rgb sub folder).

**/data/predict1/labels：** segmentation results on the simulated InSAR building images test set (The rgb display results are in ../rgb sub folder).

### Model
**/saved_models：** RefineNet_0112_163114_epoch_50.pkl