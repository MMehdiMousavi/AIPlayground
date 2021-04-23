import os

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

# --------------------------------------------------------------------------------------------
SEG = {
    'images_dir': 'UNREAL' + sep + 'image',
    'labels_dir': 'UNREAL' + sep + 'gt',
    'splits_dir': 'UNREAL' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '.png',
    'patch_shape': (388, 388),
    'window_offset': (200, 200),
    'patch_expand': (184, 184)
}
