from __future__ import  absolute_import
from __future__ import  division
import os
import json
import torch as t
from data.voc_dataset import VOCBboxDataset
from data.carrada_dataset import Carrada
from torch.utils.data import Dataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def carrada_inverse_normalize(matrix, signal_type):
    # approximate un-normalize for visualize
    if signal_type == 'range_angle':
        matrix = matrix*(152759.25 - 21.93) + 649.78
    elif signal_type == 'range_doppler':
        matrix = matrix*(95. - 13.7) + 34.39
    # return matrix.clip(min=0, max=1) * 255
    return matrix


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def carrada_normalize(matrix, signal_type):
    if signal_type == 'range_angle':
        matrix = (matrix - 649.78)/(152759.25 - 21.93)
    elif signal_type == 'range_doppler':
        matrix = (matrix - 34.39)/(95. - 13.7)
    else:
        raise TypeError('Signal type {} is not supported.'.format(signal_type))
    return matrix


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, signal_type, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
        return normalize
    else:
        return img
    # else:
        # normalize = pytorch_normalze
        # normalize = carrada_normalize


class Transform(object):

    def __init__(self, signal_type, min_size=600, max_size=1000):
        self.signal_type = signal_type
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.signal_type, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        # FLAG: data augmentation
        """
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])
        """
        return img, bbox, label, scale


"""
class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)
"""


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)


class SequenceCarradaDataset(Dataset):
    """DataLoader class for Carrada sequences
    Only shuffle sequences
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.seq_names = list(self.dataset.keys())

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        return seq_name, self.dataset[seq_name]


class CarradaDataset(Dataset):
    """DataLoader class for Carrada sequences
    Load frames
    """

    RD_SHAPE = (256, 64)
    RA_SHAPE = (256, 256)
    NB_CLASSES = 3
    CARRADA_BBOX_LABEL_NAMES = (
        'pedestrian',
        'cyclist',
        'car')


    def __init__(self, opt, dataset, annotation_type, signal_type, path_to_frames):
        self.cls = self.__class__
        self.opt = opt
        self.dataset = dataset
        self.annotation_type = annotation_type
        self.signal_type = signal_type
        self.path_to_frames = path_to_frames
        self.path_to_annots = os.path.join(self.path_to_frames, 'annotations',
                                           self.annotation_type,
                                           self.signal_type + '_light.json')
        self.tsf = Transform(self.signal_type, self.opt.min_size, self.opt.max_size)
        with open(self.path_to_annots, 'r') as fp:
            self.annots = json.load(fp)
        self.label_names = self.cls.CARRADA_BBOX_LABEL_NAMES

    def __len__(self):
        """Number of frames per sequence"""
        return len(self.dataset)

    def __getitem__(self, idx):
        frame_name = self.dataset[idx][0]
        if self.signal_type == 'range_doppler':
            matrix = np.load(os.path.join(self.path_to_frames, 'range_doppler_numpy',
                                          frame_name + '.npy'))
        elif self.signal_type == 'range_angle':
            matrix = np.load(os.path.join(self.path_to_frames, 'range_angle_numpy',
                                          frame_name + '.npy'))
        else:
            raise TypeError('Signal type {} is not supported'.format(self.signal_type))
        matrix = np.expand_dims(matrix, axis=0)
        # matrix = t.from_numpy(matrix)
        n_objets = len(self.annots[frame_name]['boxes'])
        is_empty = self.annots[frame_name]['boxes'][0] == []
        # boxes = t.FloatTensor(self.annots[frame_name]['boxes'])
        boxes = np.array(self.annots[frame_name]['boxes'])
        # labels = t.LongTensor(self.annots[frame_name]['labels'])
        labels = np.array(self.annots[frame_name]['labels'])
        difficulties = [int(is_empty)]*n_objets
        # difficulties = t.ByteTensor(difficulties)
        difficulties = np.array(difficulties)
        matrix, boxes, labels, scale = self.tsf((matrix, boxes, labels))
        # return matrix, boxes, labels, difficulties
        labels = labels - 1 # Discard the background class
        return matrix.copy(), boxes.copy(), labels.copy(), scale

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects,
        we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        matrices = list()
        boxes = list()
        labels = list()
        difficulties = list()
        for b in batch:
            matrices.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
            images = torch.stack(matrices, dim=0)
        return matrices, boxes, labels, difficulties


class TestCarradaDataset(Dataset):
    """DataLoader class for Carrada sequences
    Load frames
    """

    RD_SHAPE = (256, 64)
    RA_SHAPE = (256, 256)
    NB_CLASSES = 3

    def __init__(self, opt, dataset, annotation_type, signal_type, path_to_frames):
        self.cls = self.__class__
        self.opt = opt
        self.dataset = dataset
        self.annotation_type = annotation_type
        self.signal_type = signal_type
        self.path_to_frames = path_to_frames
        self.path_to_annots = os.path.join(self.path_to_frames, 'annotations',
                                           self.annotation_type,
                                           self.signal_type + '_light.json')
        self.tsf = Transform(self.signal_type, self.opt.min_size, self.opt.max_size)
        with open(self.path_to_annots, 'r') as fp:
            self.annots = json.load(fp)

    def __len__(self):
        """Number of frames per sequence"""
        return len(self.dataset)

    def __getitem__(self, idx):
        frame_name = self.dataset[idx][0]
        if self.signal_type == 'range_doppler':
            matrix = np.load(os.path.join(self.path_to_frames, 'range_doppler_numpy',
                                          frame_name + '.npy'))
        elif self.signal_type == 'range_angle':
            matrix = np.load(os.path.join(self.path_to_frames, 'range_angle_numpy',
                                          frame_name + '.npy'))
        else:
            raise TypeError('Signal type {} is not supported'.format(self.signal_type))
        org_matrix = np.expand_dims(matrix, axis=0)
        n_objets = len(self.annots[frame_name]['boxes'])
        is_empty = self.annots[frame_name]['boxes'][0] == []
        boxes = np.array(self.annots[frame_name]['boxes'])
        labels = np.array(self.annots[frame_name]['labels'])
        difficulties = [int(is_empty)]*n_objets
        difficulties = np.array(difficulties)

        labels = labels - 1 # Discard the background class

        if len(matrix.shape) < 3:
            matrix = np.expand_dims(matrix, axis=0)
        matrix = preprocess(matrix, self.signal_type)
        return matrix, org_matrix.shape[1:], boxes, labels, difficulties

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects,
        we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        matrices = list()
        boxes = list()
        labels = list()
        difficulties = list()
        for b in batch:
            matrices.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
            images = torch.stack(matrices, dim=0)
        return matrices, boxes, labels, difficulties
