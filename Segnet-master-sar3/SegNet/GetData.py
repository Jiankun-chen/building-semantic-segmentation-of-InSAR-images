import os
import random
import numpy as np
import imageio
import glob
from tqdm import tqdm
from tqdm import trange

class GetData:
    def __init__(self, data_dir):
        file_list = []
        images_list = []
        labels_list = []

        self.source_list = []

        examples = 0
        print("loading images")
        label_dir = os.path.join(data_dir, "Labels")
        image_master_dir = os.path.join(data_dir, "Images", "master")
        image_slave_dir = os.path.join(data_dir, "Images", "slave")
        ang_dir = os.path.join(data_dir, "Images", "ang")

        for label_root, dir, files in os.walk(label_dir):
            for file in files:
                if not file.endswith((".png", ".jpg", ".gif", "tif")):
                    continue
                try:
                    folder = os.path.relpath(label_root, label_dir)
                    image_root_master = os.path.join(image_master_dir, folder)
                    image_root_slave = os.path.join(image_slave_dir, folder)
                    ang_root = os.path.join(ang_dir, folder)

                    image_master = imageio.imread(os.path.join(image_root_master, file))
                    image_slave = imageio.imread(os.path.join(image_root_slave, file))
                    ang = imageio.imread(os.path.join(ang_root, file))

                    # image = np.array(Image.fromarray(image).resize((256, 256)))
                    # image = scipy.misc.imresize(image, 0.5)
                    label = imageio.imread(os.path.join(label_root, file))
                    # label = np.array(Image.fromarray(label).resize((256, 256)))
                    # label = scipy.misc.imresize(label, 0.5)

                    image_3 = np.stack((image_master[..., 0], image_slave[..., 0], ang[..., 0]), axis=2)

                    images_list.append(image_3)
                    labels_list.append((label[..., 0]).astype(np.int64))

                    examples = examples + 1
                except Exception as e:
                    print(e)
        print("finished loading images")
        self.examples = examples
        print("Number of examples found:", examples)
        self.images = np.array(images_list)
        self.labels = np.array(labels_list)

    def next_batch(self, batch_size):

        if len(self.source_list) < batch_size:
            new_source = list(range(self.examples))
            random.shuffle(new_source)
            self.source_list.extend(new_source)

        examples_idx = self.source_list[:batch_size]
        del self.source_list[:batch_size]

        return self.images[examples_idx, ...], self.labels[examples_idx, ...]
