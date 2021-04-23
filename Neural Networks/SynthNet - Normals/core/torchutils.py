import os
import sys
from inspect import getsource as src

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import Dataset

from core import nnviz as viz

sep = os.sep


class NNDataset(Dataset):
    def __init__(self, **kw):
        self.transforms = kw['transforms']
        self.indices = kw.get('indices', [])
        self.limit = kw.get('limit', float('inf'))
        self.mode = kw.get('mode', 'init')
        self.images_dir = kw.get('images_dir')
        self.labels_dir = kw.get('labels_dir')
        self.masks_dir = kw.get('masks_dir')
        self.parent = kw.get('parent', None)
        self.mappings = kw.get('mappings', {})

    def load_indices(self, **kw):
        raise NotImplementedError('Must be implemented by child class.')

    def __getitem__(self, index):
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    @classmethod
    def _load(cls, shuffle=False, mode=None, transforms=None, images=None, **data_conf):
        dataset = cls(mode=mode, transforms=transforms, **data_conf)
        dataset.load_indices(shuffle=shuffle, images=images if images else os.listdir(dataset.images_dir))
        return dataset

    @classmethod
    def get_loader(cls, shuffle=False, mode=None, transforms=None, images=None, run_conf=None, data_conf=None):
        dataset = cls._load(shuffle=shuffle, mode=mode, transforms=transforms, images=images, **data_conf)
        run_conf['shuffle'] = shuffle
        return NNDataLoader.get_loader(dataset=dataset, **run_conf)

    @classmethod
    def split_and_load(cls, images=None, split_ratio=(0.7, 0.15, 0.15), transforms=None, shuffle=False, conf=None,
                       run_conf=None):
        full = cls._load(shuffle=shuffle, transforms=transforms, images=images, **run_conf)
        ix = np.arange(len(full))

        if len(split_ratio) == 2:
            ix_one, ix_two = np.split(ix, [int(split_ratio[0] * len(full))])
            d1 = cls(indices=[full.indices[i] for i in ix_one], transforms=transforms, mode='train', **run_conf)
            d2 = cls(indices=[full.indices[i] for i in ix_two], transforms=transforms, mode='validation', **run_conf)
            return NNDataLoader.get_loader(dataset=d1, conf=conf), NNDataLoader.get_loader(dataset=d2, **conf)

        elif len(split_ratio) == 3:
            offset = split_ratio[0] + split_ratio[1]
            ix_one, ix_two, ix_three = np.split(ix, [int(split_ratio[0] * len(full)), int(offset * len(full))])
            d1 = cls(indices=[full.indices[i] for i in ix_one], transforms=transforms, mode='train', **run_conf)
            d2 = cls(indices=[full.indices[i] for i in ix_two], transforms=transforms, mode='validation', **run_conf)
            d3 = cls(indices=[full.indices[i] for i in ix_three], transforms=transforms, mode='test', **run_conf)
            l1 = NNDataLoader.get_loader(dataset=d1, **conf)
            l2 = NNDataLoader.get_loader(dataset=d2, **conf)
            l3 = NNDataLoader.get_loader(dataset=d3, **conf)
            return l1, l2, l3

        else:
            return ValueError(
                'Split ratio must be a ratio (r1, r2) or(r1, r2, r3) summed to 1.0. Eg. (0.8, 0.2)')


def safe_collate(batch):
    return default_collate([b for b in batch if b])


class NNDataLoader(DataLoader):

    def __init__(self, **kw):
        super(NNDataLoader, self).__init__(**kw)

    @classmethod
    def get_loader(cls, **kw):
        _kw = {
            'dataset': None,
            'batch_size': 1,
            'shuffle': False,
            'sampler': None,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
            if _kw[k]:
                print(k, ':', _kw[k])
        return cls(collate_fn=safe_collate, **_kw)


class NNTrainer:

    def __init__(self, run_conf=None, model=None, optimizer=None):

        # Initialize parameters and directories before-hand so that we can clearly track which ones are used
        self.conf = run_conf
        self.epochs_begin = 1
        self.epochs = self.conf.get('epochs', 100)
        self.log_frequency = self.conf.get('log_frequency', 10)
        self.validation_frequency = self.conf.get('validation_frequency', 1)
        self.mode = self.conf.get('mode', 'test')

        # Logging
        self.log_headers = self.get_log_headers()
        self.log_dir = self.conf.get('log_dir', 'net_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_key = self.conf.get('log_key', 'chk').split('.')[0]
        self.test_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, self.log_key + '-TEST.csv'),
                                                header=self.log_headers.get('test', ''))
        if self.mode == 'train':
            self.train_logger = NNTrainer.get_logger(
                log_file=os.path.join(self.log_dir, self.log_key + '-TRAIN.csv'),
                header=self.log_headers.get('train', ''))
            self.val_logger = NNTrainer.get_logger(log_file=os.path.join(self.log_dir, self.log_key + '-VAL.csv'),
                                                   header=self.log_headers.get('validation', ''))

        # Handle gpu/cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda" if self.conf.get('use_gpu', False) else "cpu")
        else:
            print('### GPU not found.')
            self.device = torch.device("cpu")

        # Extra utility parameters
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.patience = self.conf.get('patience', 31)
        self.checkpoint = {'latest_epoch': 0, 'latest_score': 0.0, 'latest_model_state': None,
                           'latest_optimizer_state': None, 'best_epoch': 0, 'best_score': float('inf'),
                           'best_model_state': None,
                           'best_optimizer_state': None, 'model': str(self.model),
                           'conf': str([src(v).replace(' ', '') if callable(v) else f'{str(p)}={str(v)}' for p, v in
                                        self.conf.items()])}

    def train(self, train_loader=None, validation_loader=None):
        print('Training...')

        for epoch in range(self.epochs_begin, self.epochs_begin + self.epochs + 1):
            self.model.train()
            self._adjust_learning_rate(epoch=epoch)

            self.one_epoch_run(epoch=epoch, data_loader=train_loader, logger=self.train_logger)
            self._on_epoch_end(data_loader=train_loader, log_file=self.train_logger.name)

            # Validation_frequency is the number of epoch until validation
            if epoch % self.validation_frequency == 0:
                print('----------------- Running validation... --------------------')
                self.model.eval()
                with torch.no_grad():
                    direction, score = self.validation(epoch=epoch, data_loader=validation_loader,
                                                       logger=self.val_logger)
                    self._on_validation_end(data_loader=validation_loader, log_file=self.val_logger.name)
                    self.save_checkpoint(epoch=epoch, score=score, direction=direction)
                    if self.checkpoint['latest_epoch'] - self.checkpoint['best_epoch'] >= self.patience:
                        print(f'### Patience exceeded. Stopping...')
                        return
                print('-------------------------------------------------------------')

        if not self.train_logger and not self.train_logger.closed:
            self.train_logger.close()

        if not self.val_logger and not self.val_logger.closed:
            self.val_logger.close()

    def validation(self, **kw):
        return self.one_epoch_run(epoch=kw['epoch'], data_loader=kw['data_loader'],
                                  logger=kw['logger'])

    def test(self, **kw):
        raise NotImplementedError('Must be implemented by a child class.')

    def _on_epoch_end(self, **kw):
        viz.plot_column_keys(file=kw['log_file'], batches_per_epoch=kw['data_loader'].__len__(),
                             keys=['F1', 'LOSS', 'ACCURACY'], title='Train')
        viz.plot_cmap(file=kw['log_file'], save=True, x='PRECISION', y='RECALL', title='Train')

    def _on_validation_end(self, **kw):
        viz.plot_column_keys(file=kw['log_file'], batches_per_epoch=kw['data_loader'].__len__(),
                             keys=['F1', 'ACCURACY'], title='Validation')
        viz.plot_cmap(file=kw['log_file'], save=True, x='PRECISION', y='RECALL', title='Validation')

    def _on_test_end(self, **kw):
        viz.y_scatter(file=kw['log_file'], y='F1', label='ID', save=True, title='Test')
        viz.y_scatter(file=kw['log_file'], y='ACCURACY', label='ID', save=True, title='Test')
        viz.xy_scatter(file=kw['log_file'], save=True, x='PRECISION', y='RECALL', label='ID', title='Test')

    # Headers for log files
    def get_log_headers(self):
        return {
            'train': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'validation': 'ID,EPOCH,BATCH,PRECISION,RECALL,F1,ACCURACY,LOSS',
            'test': 'ID,PRECISION,RECALL,F1,ACCURACY,LOSS'
        }

    def resume_from_checkpoint(self, checkpoint_file=None, parallel_trained=False, key=None):
        try:
            path = checkpoint_file if checkpoint_file else os.path.join(self.log_dir, f'{self.log_key}.tar')
            checkpoint = torch.load(path, map_location=self.device)
            print(path, 'Loaded...')
            if parallel_trained:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint[f'{key}_model_state'].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)
                self.optimizer.load_state_dict(checkpoint[f'{key}_optimizer_state'])
            else:
                self.model.load_state_dict(checkpoint[f'{key}_model_state'])
                self.optimizer.load_state_dict(checkpoint[f'{key}_optimizer_state'])
            self.epochs_begin = checkpoint[f'{key}_epoch']
        except Exception as e:
            print('ERROR: ' + str(e))

    def save_checkpoint(self, **kw):

        assert (kw['direction'] == 'maximize' or kw[
            'direction'] == 'minimize'), 'direction must be minimize(eg. MSE) or maximize(eg. F1 score)'

        self.checkpoint['latest_model_state'] = self.model.state_dict()
        self.checkpoint['latest_optimizer_state'] = self.optimizer.state_dict()
        self.checkpoint['latest_epoch'] = kw.get('epoch')
        self.checkpoint['latest_score'] = kw.get('score')

        self.checkpoint['best_score'] = self.checkpoint.get('best_score', kw.get('score'))
        self.checkpoint['best_epoch'] = self.checkpoint.get('best_epoch', kw.get('epoch'))
        self.checkpoint['best_model_state'] = self.checkpoint.get('best_model_state', self.model.state_dict())
        self.checkpoint['best_optimizer_state'] = self.checkpoint.get('best_optimizer_state',
                                                                      self.optimizer.state_dict())
        if kw['direction'] == 'maximize' and kw['score'] >= self.checkpoint['best_score'] \
                or kw['direction'] == 'minimize' and kw['score'] <= self.checkpoint['best_score']:
            print(
                f'#### SCORE IMPROVED from {self.checkpoint["best_score"]} to {self.checkpoint["latest_score"]}.')
            self.checkpoint['best_score'] = self.checkpoint["latest_score"]
            self.checkpoint['best_epoch'] = self.checkpoint["latest_epoch"]
            self.checkpoint['best_model_state'] = self.checkpoint['latest_model_state']
            self.checkpoint['best_optimizer_state'] = self.checkpoint['latest_optimizer_state']
        else:
            print(
                f'Score did not improve. Best was {self.checkpoint["best_score"]} on epoch {self.checkpoint["best_epoch"]}.')

        torch.save(self.checkpoint, os.path.join(self.log_dir, f'{self.log_key}.tar'))

    @staticmethod
    def get_logger(log_file=None, header=''):
        if os.path.isfile(log_file):
            print('### CRITICAL!!! ' + log_file + '" already exists.')
            ip = input('Override? [Y/N]: ')
            if ip == 'N' or ip == 'n':
                sys.exit(1)

        file = open(log_file, 'w')
        NNTrainer.flush(file, header)
        return file

    @staticmethod
    def flush(logger, msg):
        if logger is not None:
            logger.write(msg + '\n')
            logger.flush()

    def _adjust_learning_rate(self, epoch):
        if epoch % 30 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] >= 1e-5:
                    param_group['lr'] = param_group['lr'] * 0.7

    def one_epoch_run(self, **kw):
        raise NotImplementedError('Must be implemented.')
