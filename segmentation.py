import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG
from easytorch import ETTrainer, ETDataset, ConfusionMatrix
from easytorch.vision import (Image, expand_and_mirror_patch, merge_patches, get_chunk_indexes)

from models import UNet

sep = os.sep

CLASS_LABELS = {
    'Other': 0,
    'Table': 1,
    'Shelve': 2,
    'Book': 3,
    'Couch': 4,
    'Frame': 5,
    'Lamp': 6,
    'Plant': 7,
    'Rug': 8,
    'Wall': 9,
    'Floor': 10,
    'TV': 11,
    'Window': 12,
    'Curtain': 13,
    'Door': 14}
CLASS_RGB = {
    'Other': (0, 0, 0),
    'Table': (89, 20, 0),
    'Shelve': (0, 25, 0),
    'Book': (38, 51, 76),
    'Couch': (0, 0, 51),
    'Frame': (255, 0, 0),
    'Lamp': (101, 178, 0),
    'Plant': (255, 255, 0),
    'Rug': (0, 255, 0),
    'Wall': (101, 101, 101),
    'Floor': (0, 255, 255),
    'TV': (255, 0, 76),
    'Window': (204, 204, 153),
    'Curtain': (0, 76, 255),
    'Door': (255, 0, 255)}

num_classes = len(CLASS_LABELS)


class SEGDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)

    def load_index(self, dataset_name, file):
        D = self.dataspecs[dataset_name]
        img_obj = Image()
        img_obj.load(D['data_dir'], file)
        img_obj.load_ground_truth(D['label_dir'], D['label_getter'])
        img_obj.apply_clahe()

        gt = np.zeros_like(img_obj.array[:, :, 0])
        for cls in list(CLASS_LABELS.keys()) \
                if self.args['classification_mode'].upper() == 'MULTI' \
                else [self.args['classification_mode']]:
            r, g, b = CLASS_RGB[cls]
            logic = np.logical_and.reduce(
                [img_obj.ground_truth[:, :, 0] == r, img_obj.ground_truth[:, :, 1] == g,
                 img_obj.ground_truth[:, :, 2] == b])
            ix = np.array(list(zip(*np.where(logic))))
            if len(ix) > 0:
                gt[ix[:, 0], ix[:, 1]] = CLASS_LABELS[cls]
            else:
                print(f'### {img_obj.file} has no {cls}.')
        img_obj.ground_truth = gt

        self.data[file] = img_obj
        for chunk_ix in get_chunk_indexes(img_obj.array.shape[0:2], D['patch_shape'], D['patch_offset']):
            self.indices.append([dataset_name, file] + chunk_ix)

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
        return {'indices': self.indices[index], 'input': arr3, 'label': gt.copy()}

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.ToPILImage(), tmf.ToTensor()])


class SEGTrainer(ETTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.num_class = len(CLASS_LABELS) if self.args['classification_mode'].upper() == 'MULTI' else 2
        print(self.num_class)

    def _init_nn_model(self):
        self.nn['model'] = UNet(self.args['num_channel'], self.num_class, reduce_by=self.args['model_scale'])

    def iteration(self, batch):
        inputs = batch['input'].to(self.device['gpu']).float()
        labels = batch['label'].to(self.device['gpu']).long()

        out = self.nn['model'](inputs)
        out = F.softmax(out, 1)
        loss = F.nll_loss(F.log_softmax(out, 1), labels)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels)

        avg = self.new_averages()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}

    def save_predictions(self, dataset, its):
        dataset_name = list(dataset.dataspecs.keys())[0]
        D = dataset.dataspecs[dataset_name]
        file = list(dataset.data.values())[0].file
        img_shape = dataset.data[file].array.shape[:2]

        patches = its['output']()[:, 1, :, :].cpu().numpy() * 255
        img = merge_patches(patches, img_shape, D['patch_shape'], D['patch_offset'])
        IMG.fromarray(img).save(self.cache['log_dir'] + sep + dataset_name + '_' + file)

    def init_experiment_cache(self):
        self.cache.update(monitor_metric='f1', metric_direction='maximize')
        self.cache.update(log_header='Loss|Accuracy,F1,Precision,Recall')

    def new_metrics(self):
        return ConfusionMatrix(num_classes=self.num_class)
