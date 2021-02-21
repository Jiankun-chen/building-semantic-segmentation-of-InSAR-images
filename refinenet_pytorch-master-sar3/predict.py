import os
import time
from PIL import Image
import torch as t
from config import DefaultCofig as cfg
from net.refinenet.refinenet_4cascade import RefineNet4Cascade
from torch.utils import data
import utils
import dataset

load_start = time.time()

# load net
net = RefineNet4Cascade(input_shape=(3, 256, 256), num_classes=3)
if cfg.use_gpu:
    net.cuda()

net.load_state_dict(t.load(cfg.test_model))

predict_data = dataset.PredictINputDataset(cfg.predict_images_ang, cfg.predict_images_master, cfg.predict_images_slave)
predict_dataLoader = data.DataLoader(predict_data, batch_size=1)

load_end = time.time()
print('load model time: {0}ms'.format(1000 * (load_end - load_start)))

start = time.time()
with t.no_grad():
    for i, (x1, x2, x3, name) in enumerate(predict_dataLoader):

        cur_start = time.time()

        if cfg.use_gpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
        xx = t.cat((x1, x2, x3), 1)
        xxx = t.cat((xx[:, 0:1, :, :], xx[:, 3:4, :, :], xx[:, 6:7, :, :]), 1)

        y1_, y2_ = net(xxx)

        seg = utils.seg_transfer(y1_[0].cpu().numpy())
        depth = utils.depth_transfer(y2_[0][0].cpu().numpy(), 255)


        seg = Image.fromarray(seg.astype('uint8'))
        # depth = Image.fromarray(depth.astype('uint8'))
        #
        seg.save(os.path.join(cfg.predict_labels, name[0] + '.png'))
        # depth.save(os.path.join(cfg.predict_depths, name[0] + '.png'))

        cur_end = time.time()
        print('img {0} USED: {1}ms, predict total: {2}ms'.format(i + 1, 1000 * (cur_end - cur_start),
                      1000 * (cur_end - start)))