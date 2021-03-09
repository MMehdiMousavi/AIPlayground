import os

sep = os.sep
# --------------------------------------------------------------------------------------------
NORM = {
    'images_dir': 'UNREAL' + sep + 'image',
    'labels_dir': 'UNREAL' + sep + 'normal',
    'splits_dir': 'UNREAL' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '.png',
    'patch_shape': (388, 388),
    'window_offset': (200, 200),
    'patch_expand': (184, 184)
}
