#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

sep = os.sep
import torchvision.transforms as tmf
import torch.nn.functional as F

from core.measurements import NNVal
from core.torchutils import NNTrainer, NNDataset
from core import image_utils as iu
from core.image_utils import Image
from PIL import Image as IMG
import core.datautils as du

import numpy as np

import random
import os
import traceback

import torch
import torch.optim as optim
from models import UNet
import argparse
import runs


class KernelDataset(NNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_label = kw.get('label_getter')
        self.patch_shape = kw.get('patch_shape')
        self.patch_stride = kw.get('patch_stride')
        self.expand_by = kw.get('patch_expand_by')
        self.input_size = (self.patch_shape[0] + self.expand_by[0], self.patch_shape[1] + self.expand_by[1])

    def load_indices(self, images=None, shuffle=False):
        print(self.images_dir, '...')

        for fc, file in enumerate(images, 1):
            print(f'{file}, {fc}', end='\r')
            img_obj = Image()
            img_obj.load(self.images_dir, file)
            img_obj.load_ground_truth(self.labels_dir, self.get_label)
            img_obj.apply_clahe()

            img_obj.array = img_obj.array[:, :, 0:3]
            img_obj.ground_truth = img_obj.ground_truth[:, :, 0:3]

            for chunk_ix in iu.get_chunk_indexes(img_obj.array.shape[0:2], self.patch_shape, self.patch_stride):
                self.indices.append([fc] + chunk_ix)
                self.mappings[fc] = img_obj

            if len(self) >= self.limit:
                break

        if shuffle:
            random.shuffle(self.indices)
        print(f'{len(self)} Indices Loaded')

    def __getitem__(self, index):
        ID, p, q, r, s = self.indices[index]
        img = self.mappings[ID]
        arr = img.array
        gt = img.ground_truth[p:q, r:s, :]
        gt[gt == 255] = 1

        _p, _q, _r, _s, pad = iu.expand_and_mirror_patch(full_img_shape=(arr.shape[0], arr.shape[1]),
                                                         orig_patch_indices=[p, q, r, s],
                                                         expand_by=self.expand_by)

        arr3 = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        arr3[:, :, 0] = np.pad(arr[_p:_q, _r:_s, 0], pad, 'reflect')
        arr3[:, :, 1] = np.pad(arr[_p:_q, _r:_s, 1], pad, 'reflect')
        arr3[:, :, 2] = np.pad(arr[_p:_q, _r:_s, 2], pad, 'reflect')

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            arr3 = np.flip(arr3.copy(), 0)
            gt = np.flip(gt.copy(), 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            arr3 = np.flip(arr3.copy(), 1)
            gt = np.flip(gt.copy(), 1)

        if self.transforms is not None:
            arr3 = self.transforms(IMG.fromarray(arr3))
            gt = self.transforms(IMG.fromarray(gt))

        return {'indices': index, 'inputs': arr3, 'labels': gt}


class KernelTrainer(NNTrainer):
    def __init__(self, **kw):
        super(KernelTrainer, self).__init__(**kw)

    # Headers for log files
    def get_log_headers(self):
        return {
            'train': 'ID,EPOCH,BATCH,LOSS',
            'validation': 'ID,EPOCH,LOSS',
            'test': 'ID,LOSS'
        }

    def test(self, data_loader=None, global_score=None, logger=None):
        print('------Running test------')
        score = NNVal()
        self.model.eval()
        img_objects = {}
        with torch.no_grad():
            for i, data in enumerate(data_loader, 1):
                inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()
                indices = data['indices'].to(self.device).long()
                outputs = self.model(inputs)
                loss = F.smooth_l1_loss(outputs, labels)
                score.add(loss.item())
                print(f'Batch: {i}/{len(data_loader)}', end='\r')
                for ix, pred in enumerate(outputs):
                    obj_id, p, q, r, s = data_loader.dataset.indices[indices[ix].item()]
                    if not img_objects.get(obj_id):
                        img_objects[obj_id] = data_loader.dataset.mappings[obj_id]

                    arr = np.rollaxis(outputs[ix].cpu().numpy(), 0, 3)
                    if img_objects.get(obj_id).extras.get('mosaic') is not None:
                        img_objects.get(obj_id).extras.get('mosaic').append([p, q, r, s, arr])
                    else:
                        img_objects.get(obj_id).extras['mosaic'] = [[p, q, r, s, arr]]

        global_score.accumulate(score)
        for ID, obj in img_objects.items():
            arr = np.zeros_like(obj.array, dtype=np.uint8)
            for p, q, r, s, _arr in obj.extras['mosaic']:
                arr[p:q, r:s] = _arr * 255

            IMG.fromarray(arr).save(self.log_dir + os.sep + obj.file.split('.')[0] + '_pred.png')
        self.flush(logger, ','.join(str(x) for x in ['AGGREGATE', score.average]))

        if not logger.closed:
            logger.close()

    def one_epoch_run(self, **kw):
        """
        One epoch implementation of binary cross-entropy loss
        :param kw:
        :return:
        """
        metrics = NNVal()
        running_loss = NNVal()
        data_loader = kw['data_loader']
        for i, data in enumerate(data_loader, 1):
            inputs, labels = data['inputs'].to(self.device).float(), data['labels'].to(self.device).float()

            if self.model.training:
                self.optimizer.zero_grad()

            loss = F.smooth_l1_loss(self.model(inputs), labels)
            current_loss = loss.item()
            running_loss.add(current_loss)

            if self.model.training:
                loss.backward()
                self.optimizer.step()
                metrics.reset()

            metrics.add(current_loss)
            if i % self.log_frequency == 0:
                print('Epochs[%d/%d] Batch[%d/%d] SmoothL1:%.5f' % (
                    kw['epoch'], self.epochs, i, len(kw['data_loader']), running_loss.average))
                running_loss.reset()

            self.flush(kw['logger'],
                       ','.join(str(x) for x in [0, kw['epoch'], i, current_loss]))
        return 'minimize', metrics.average


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


ap = argparse.ArgumentParser()
ap.add_argument("-nch", "--input_channels", default=3, type=int, help="Number of channels of input image.")
ap.add_argument("-ncl", "--num_classes", default=3, type=int, help="Number of output classes.")
ap.add_argument("-b", "--batch_size", default=32, type=int, help="Mini batch size.")
ap.add_argument('-ep', '--epochs', default=51, type=int, help='Number of epochs.')
ap.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate.')
ap.add_argument('-gpu', '--use_gpu', default=True, type=boolean_string, help='Use GPU?')
ap.add_argument('-d', '--distribute', default=True, type=boolean_string, help='Distribute to all GPUs.')
ap.add_argument('-s', '--shuffle', default=True, type=boolean_string, help='Shuffle before each epoch.')
ap.add_argument('-lf', '--log_frequency', default=3, type=int, help='Log after ? iterations.')
ap.add_argument('-vf', '--validation_frequency', default=1, type=int, help='Validation after ? epochs.')
ap.add_argument('-pt', '--parallel_trained', default=False, type=boolean_string,
                help='If model to resume was parallel trained.')
ap.add_argument('-pin', '--pin_memory', default=True, type=boolean_string,
                help='Pin Memory.')
ap.add_argument('-nw', '--num_workers', default=0, type=int, help='Number of workers to work with data loading.')
ap.add_argument('-chk', '--checkpoint_file', default=None, type=str, help='Name of the checkpoint file.')
ap.add_argument('-m', '--mode', required=True, type=str, help='Mode of operation.')
ap.add_argument('-data', '--data_dir', default='data', required=False, type=str, help='Root path to input Data.')
ap.add_argument('-lbl', '--label', type=str, nargs='+', help='Label to identify the experiment.')
ap.add_argument('-lim', '--load_limit', default=float('inf'), type=int, help='Data load limit')
ap.add_argument('-log', '--log_dir', default='net_logs', type=str, help='Logging directory.')
run_conf = vars(ap.parse_args())

transforms = tmf.Compose([
    tmf.ToTensor()
])

test_transforms = tmf.Compose([
    tmf.ToTensor()
])


def run(conf, data):
    score = NNVal()
    for file in os.listdir(data['splits_dir']):
        conf['log_key'] = file
        split = du.load_split_json(data['splits_dir'] + sep + file)
        model = UNet(conf['input_channels'], conf['num_classes'], reduce_by=2)
        optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'])

        if conf['distribute']:
            model = torch.nn.DataParallel(model)
            model.float()
            optimizer = optim.Adam(model.module.parameters(), lr=conf['learning_rate'])

        try:
            trainer = KernelTrainer(run_conf=conf, model=model, optimizer=optimizer)

            if conf.get('mode') == 'train':
                train_loader = KernelDataset.get_loader(shuffle=True, mode='train', transforms=transforms,
                                                        images=split['train'], data_conf=data,
                                                        run_conf=conf)

                validation_loader = KernelDataset.get_loader(shuffle=True, mode='validation',
                                                             transforms=transforms,
                                                             images=split['validation'], data_conf=data,
                                                             run_conf=conf)

                print('### Train Val Batch size:', len(train_loader), len(validation_loader))
                # trainer.resume_from_checkpoint(parallel_trained=conf.get('parallel_trained'), key='latest')
                trainer.train(train_loader=train_loader, validation_loader=validation_loader)

            test_loader = KernelDataset.get_loader(shuffle=False, mode='test', transforms=transforms,
                                                   images=split['test'], data_conf=data, run_conf=conf)

            trainer.resume_from_checkpoint(checkpoint_file=conf.get("checkpoint_file"),
                                           parallel_trained=conf.get('parallel_trained'), key='best')
            trainer.test(data_loader=test_loader, global_score=score, logger=trainer.test_logger)
        except Exception as e:
            traceback.print_exc()

    with open(conf.get('log_dir', 'net_logs') + os.sep + 'global_score.txt', 'w') as lg:
        lg.write(f'{score.average}')
        lg.flush()


"""
################# -Example- ######################################
---NORMAL---
###################################################################
"""
from runs import NORM

data_confs = [NORM]
if __name__ == "__main__":
    for data_conf in data_confs:
        for k, v in data_conf.items():
            if 'dir' in k:
                data_conf[k] = run_conf['data_dir'] + os.sep + data_conf[k]

        data_conf['patch_shape'] = (388, 388)
        data_conf['patch_stride'] = (200, 200)
        data_conf['patch_expand_by'] = (184, 184)
        run(run_conf, data_conf)
