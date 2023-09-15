import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

import pdb

class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
        - **get_ood** (bool, optional) : If True the ood data of the correspoonding split is returned
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'ood_id','category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0,   255, 'flat', 1, False, False, (128, 64, 128)), ##
        CityscapesClass('sidewalk',             8, 1,   255, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,  255, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,  255, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,  255, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255,  255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255,  255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 255, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 255, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 255, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 255, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 255, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 255, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 255, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 255, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 255, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 255, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 255, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 255, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 255, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 255, 'vehicle', 7, True, False, (119, 11, 32)), ## changin 18 to 255 to see if this trains
        CityscapesClass('license plate',        -1, 255, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = []
    train_ood_id_to_color = []
    id_to_train_id = []
    id_to_train_ood_id = []
    total_id_classes = 0
    total_ood_classes = 0
    updated = False

    # train_id_to_color.append([0, 0, 0])
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.train_id for c in classes])
    #


    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None, get_ood = False, ood_classes = None, ood_id = None):

        ood_classes = ood_classes
        ood_classes_id = ood_id
        if not self.updated:
            self.update_classes(ood_classes_id)


        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.get_ood = get_ood
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        #
        self.targets_id = []
        self.images_id = []
        self.targets_ood = []
        self.images_ood = []

        if ood_classes_id is not None:
            self.create_ood_id_split(ood_classes_id)
        else:
            print("No OOD classes")
            self.targets_id = self.targets
            self.images_id = self.images

    def create_ood_id_split(self, ood_classes_id):

        ood_classes_id = np.array(ood_classes_id)

        for i in range(0, len(self.targets)):

            target = Image.open(self.targets[i])

            if np.any(np.isin(np.array(target), ood_classes_id)):
                _, target_resized = self.transform(Image.open(self.images[i]).convert('RGB'),target)
                if np.any(np.isin(np.array(target_resized), ood_classes_id)):
                    self.targets_ood.append(self.targets[i])
                    self.images_ood.append(self.images[i])
                else:
                    self.targets_id.append(self.targets[i])
                    self.images_id.append(self.images[i])

            else:
                self.targets_id.append(self.targets[i])
                self.images_id.append(self.images[i])


    @classmethod
    def update_classes(cls, ood_classes):

        ood_id = 0
        if ood_classes is not None:

            for c_idx in range(0,len(cls.classes)):

                if cls.classes[c_idx].id in ood_classes:
                    c = cls.classes[c_idx]
                    cls.classes[c_idx] = cls.CityscapesClass(c.name,c.id, 255, ood_id, c.category, c.category_id, c.has_instances, c.ignore_in_eval, c.color)
                    ood_id +=1
            ## fix train_id
        curr_id = 0
        for c_idx in range(0,len(cls.classes)):
            if cls.classes[c_idx].train_id != 255:
                c = cls.classes[c_idx]
                cls.classes[c_idx] =  cls.CityscapesClass(c.name, c.id, curr_id, c.ood_id, c.category, c.category_id, c.has_instances, c.ignore_in_eval, c.color)
                curr_id+=1
        
        cls.total_id_classes = curr_id
        cls.total_ood_classes = ood_id

        cls.train_id_to_color = [c.color for c in cls.classes if (c.train_id != -1 and c.train_id != 255)]
        cls.train_ood_id_to_color = cls.train_id_to_color.copy()

        cls.train_ood_id_to_color += [c.color for c in cls.classes if c.ood_id != 255]


        cls.train_id_to_color.append([0, 0, 0])
        cls.train_ood_id_to_color.append([0,0,0])
        cls.train_id_to_color = np.array(cls.train_id_to_color)
        cls.train_ood_id_to_color = np.array(cls.train_ood_id_to_color)

        # cls.id_to_train_id = [c.train_id for c in cls.classes]

        for c in cls.classes:

            cls.id_to_train_id += [c.train_id]
            if c.ood_id != 255:
                cls.id_to_train_ood_id += [c.ood_id + cls.total_id_classes]
            else:
                cls.id_to_train_ood_id += [c.train_id]

        cls.id_to_train_id = np.array(cls.id_to_train_id)
        cls.id_to_train_ood_id = np.array(cls.id_to_train_ood_id)

        cls.updated = True

    @classmethod
    def encode_target(cls, target, ood=False):
        if not ood:
            return cls.id_to_train_id[np.array(target)]
        else:
            return cls.id_to_train_ood_id[np.array(target)]
    @classmethod
    def decode_target(cls, target, ood = False):
        if not ood:
            target[target == 255] = cls.total_id_classes ## for ID
            #target = target.astype('uint8') + 1
            return cls.train_id_to_color[target]
        else:

            target[target == 255] = cls.total_id_classes + cls.total_ood_classes
            #target = target.astype('uint8') + 1
            return cls.train_ood_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        if self.get_ood:
            image = Image.open(self.images_ood[index]).convert('RGB')
            target = Image.open(self.targets_ood[index])

        else:
            image = Image.open(self.images_id[index]).convert('RGB')
            target = Image.open(self.targets_id[index])
        if self.transform:
            image, target = self.transform(image, target)

        target = self.encode_target(target, ood = self.get_ood)
        # target_ood = self.encode_target(target, ood=True)
        return image, target

    def __len__(self):
        if self.get_ood:
            return len(self.images_ood)
        else:
            return len(self.images_id)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
