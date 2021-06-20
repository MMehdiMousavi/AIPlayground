import argparse

"""Note: check default_ap for runtime argument usage"""
from easytorch import default_ap, EasyTorch

from normals import *

sep = os.sep


def same(x): return x


NORM_DATA = {
    'name': 'NORM',
    'data_dir': 'Picture_Caustic',
    'label_dir': 'Normal',
    # 'split_dir': 'Caustic\\splits',
    'label_getter': same,
    'patch_shape': (388, 388),
    'patch_offset': (350, 350),
    'expand_by': (184, 184)
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(parents=[default_ap], add_help=False)
    ap.add_argument('-ms', '--model_scale', default=1, type=int, help='model_scale')

    """RUN as : python normals_main.py -ph train -b 4 -e 251 -ms 1 -data datasets"""
    runner = EasyTorch(dataspecs=[NORM_DATA], args=ap, load_sparse=True, num_channel=3, num_class=3)
    runner.run(NormalsTrainer, NormalsDataset)
