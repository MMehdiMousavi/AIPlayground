import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG
from easytorch import ETTrainer, ETDataset, ETAverages
from easytorch.vision import (Image, expand_and_mirror_patch, merge_patches)
from easytorch.vision import get_chunk_indexes
from torchvision import transforms as tmf, utils as vutils

from models import UNet

sep = os.sep


class NormalsDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)

    def load_index(self, dataset_name, file):
        D = self.dataspecs[dataset_name]
        img_obj = Image()
        img_obj.load(D['data_dir'], file)
        img_obj.load_ground_truth(D['label_dir'], D['label_getter'])

        img_obj.array = img_obj.array[:, :, 0:3]
        img_obj.ground_truth = img_obj.ground_truth[:, :, 0:3]

        img_obj.apply_mask()
        img_obj.apply_clahe()
        self.data[file] = img_obj
        for p, q, r, s in get_chunk_indexes(img_obj.array.shape[0:2], D['patch_shape'], D['patch_offset']):
            self.indices.append([dataset_name, file, p, q, r, s])

    def __getitem__(self, index):
        dname, file, row_from, row_to, col_from, col_to = self.indices[index]
        D = self.dataspecs[dname]
        input_size = tuple(map(sum, zip(D['patch_shape'], D['expand_by'])))

        arr = self.data[file].array
        gt = self.data[file].ground_truth[row_from:row_to, col_from:col_to]

        p, q, r, s, pad = expand_and_mirror_patch(arr.shape, [row_from, row_to, col_from, col_to], D['expand_by'])
        arr3 = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
        arr3[:, :, 0] = np.pad(arr[p:q, r:s, 0], pad, 'reflect')
        arr3[:, :, 1] = np.pad(arr[p:q, r:s, 1], pad, 'reflect')
        arr3[:, :, 2] = np.pad(arr[p:q, r:s, 2], pad, 'reflect')

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            arr3 = np.flip(arr3, 0)
            gt = np.flip(gt, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            arr3 = np.flip(arr3, 1)
            gt = np.flip(gt, 1)

        arr3 = self.transforms(arr3)
        gt = self.transforms(gt)
        return {'indices': self.indices[index], 'input': arr3, 'label': gt}

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.ToPILImage(), tmf.ToTensor()])


class NormalsTrainer(ETTrainer):

    def _init_nn_model(self):
        self.nn['model'] = UNet(self.args['num_channel'], self.args['num_class'], reduce_by=self.args['model_scale'])

    def iteration(self, batch):
        inputs = batch['input'].to(self.device['gpu']).float()
        labels = batch['label'].to(self.device['gpu']).float()

        out = self.nn['model'](inputs)
        out = torch.sigmoid(out)
        loss = F.mse_loss(out, labels)

        sc = self.new_metrics()
        sc.add(loss.item(), len(inputs))

        avg = self.new_averages()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'averages': avg, 'metrics': sc, 'output': out, 'input': inputs}

    def save_predictions(self, dataset, its):
        dataset_name = list(dataset.dataspecs.keys())[0]
        D = dataset.dataspecs[dataset_name]

        file = list(dataset.data.values())[0].file
        img_shape = dataset.data[file].array.shape[:2]

        r = its['output']()[:, 0, :, :].cpu().numpy() * 255
        g = its['output']()[:, 1, :, :].cpu().numpy() * 255
        b = its['output']()[:, 2, :, :].cpu().numpy() * 255
        r = merge_patches(r, img_shape, D['patch_shape'], D['patch_offset'])
        g = merge_patches(g, img_shape, D['patch_shape'], D['patch_offset'])
        b = merge_patches(b, img_shape, D['patch_shape'], D['patch_offset'])
        rgb = np.rollaxis(np.array([r, g, b]), 0, 3)
        IMG.fromarray(rgb).save(self.cache['log_dir'] + sep + dataset_name + '_' + file)

    def init_experiment_cache(self):
        self.cache.update(monitor_metric='average', metric_direction='minimize')
        self.cache.update(log_header='MSE_loss')

    def new_metrics(self):
        return ETAverages()

    # def _on_iteration_end(self, **kw):
    #     if kw['i'] % 256 == 0:
    #         grid = vutils.make_grid(kw['it']['output']()[:10], padding=2, normalize=True)
    #         vutils.save_image(grid, f"{self.cache['log_dir']}{sep}recons.png")
    #         grid = vutils.make_grid(kw['it']['input']()[:10], padding=2, normalize=True)
    #         vutils.save_image(grid, f"{self.cache['log_dir']}{sep}real.png")

