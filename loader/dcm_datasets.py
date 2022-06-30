# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch

# ----------------------------------------------------------------------------
try:
    import dnnlib
except ImportError:
    dnnlib = None

# ----------------------------------------------------------------------------
## Custom imports
from utils import dcm_to_array
from utils import array_to_sitk_image
from utils import sitk_image_to_array
from utils import normalize
from utils import clip, window, window_and_stack
from utils import dynamic_range
from utils import percentile
from utils import padding, resize

from transforms import ToTensor, Normalize
from transforms import ToPILImage, SquarePad
# ----------------------------------------------------------------------------


# https://github.com/NVlabs/stylegan3/blob/main/training/dataset.py
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,
            name,                       # Name of the dataset
            raw_shape,                  # Shape of the raw image data (NCHW).
            max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
            use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
            xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
            random_seed = 0,        # Random seed to use when applying max_size.
    ):
        
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
       
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])


    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):
        raise NotImplementedError

    def _load_raw_labels(self):
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    
    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype in [np.uint8, np.int16]

        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
        image = image[:, :, ::-1]

        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        if dnnlib is not None:
            d = dnnlib.EasyDict()
            d.raw_idx = int(self._raw_idx[idx])
            d.xflip = (int(self._xflip[idx]) != 0)
            d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
            return d
        else:
            raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) in [1, 3]# CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        """
            self.resolution is not decided by raw shape,
            but by resolution (user custom argument)
        """
        assert len(self.image_shape) in [1, 3] # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]


    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64



#--------------------------------------------------------------------------------------------#


class DCMFolderDataset(Dataset):

    def __init__(self,
            path,
            resolution      = None,
            modality        = None,     ## Medical image modality
            windowing       = None,     ## Windowing option argument 
            output_dtype    = 'uint8',  ## Default data type
            output_channels = 3,
            transform       = None,
            target_transform = None,
            **super_kwargs,         # Additional arguments for the Dataset base class.
    ):

        self._path = path

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
            self._all_fnames = list(self._all_fnames)

        ##TODO: zip file support
        else:
            raise IOError('Path must point to a directory')

        ### Custom args
        self._modality  = modality
        self._windowing = windowing
        self._o_dtype   = output_dtype
        self._o_channels = output_channels

        IMG_EXTENSIONS = PIL.Image.EXTENSION.copy()
        IMG_EXTENSIONS['.dcm'] = 'DICOM'

        PIL.Image.init()

        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in IMG_EXTENSIONS)

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        _image_shape = [raw_shape[1]] + [resolution] * 2

        # Resolution will be matched after preprocessing in getitem
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            #raise IOError('Image files do not match the specified resolution')
            print(f"Detected shape is {raw_shape[1]}x{raw_shape[2]}x{raw_shape[3]}")
            print(f"Will be pre-processed to {output_channels}x{resolution}x{resolution}, dtype: {output_dtype}")

        self.transform = transform
        self.target_transform = target_transform

        super().__init__(name=name, raw_shape=[raw_shape[0]]+_image_shape, **super_kwargs)

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx]) # CHW

        assert isinstance(image, np.ndarray)
        assert image.dtype in [np.uint8, np.int16]

        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        ## 3 options
        if self._modality == 'xray':
            image = _preprocess_xray(image, self._windowing, self._o_dtype, self._o_channels)
        elif self._modality == 'headct':
            image = _preprocess_headct(image, self._windowing, self._o_dtype, self._o_channels)
        elif self._modality == 'abdomenct':
            image = _preprocess_abdomenct(image, self._windowing, self._o_dtype, self._o_channels)
        else:
            assert 0, print("Please check modality argument")

        if self.transform is not None:
            image = self.transform(image.copy())

        target = self.get_label(idx)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._file_ext(fname) == '.dcm':
                dicom = dcm_to_array(os.path.join(self._path, fname))
                image = np.array(dicom) #CHW
            else:
                image = np.array(PIL.Image.open(f)) #TODO: HWC to CHW
        if image.ndim == 2:
            image = image[np.newaxis,:, :] # HW => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        try:
            labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        except:
            labels = [labels['/'+fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        return None



def _preprocess_headct(image, windowing, o_dtype, o_channels):
    image = clip(image, center=1024, width=4096) # [-1024~3071]
    image_meta = array_to_sitk_image(image)

    if o_dtype == 'uint8':
        if o_channels == 1:
            if windowing == 'normal':
                # Brain Window WW: 40, WL:80 => [0, 80]
                image_meta = window(image_meta, w_range=(0, 80))
            elif windowing == 'hemorrhage':
                # Subdural Window WW:80, WL:200 => [-20, 180]
                image_meta = window(image_meta, w_range=(-20, 180))
        else: 
            image_meta = window_and_stack(image_meta)
    #TODO: dtype == 'int16'

    image = sitk_image_to_array(image_meta)

    if o_channels == 3:
        image = np.transpose(image[0], (2, 0, 1)) # HWC -> CHW
    return image 


def _preprocess_abdomenct(image, windowing, o_dtype, o_channels):
    image = clip(image, center=1024, width=4096) # [-1024~3071]
    image_meta = array_to_sitk_image(image)

    if o_dtype == 'uint8':
        if o_channels == 1:
            if windowing == 'normal':
                # WL 35, WW 350 => [-140 ~ 210]
                image_meta = window(image_meta, w_range=(-140, 210))
            elif windowing == 'hemoperitoneum':
                # WL:20 WW:200 => [-80 ~ 120]
                image_meta = window(image_meta, w_range=(-80, 120))
        else:
            image_meta = window_and_stack(image_meta)

    image = sitk_image_to_array(image_meta)
    if o_channels == 3:
        image = np.transpose(image[0], (2, 0, 1)) # HWC -> CHW

    ##TODO: 12-bit training

    return image


def _preprocess_xray(image, windowing, o_dtype, o_channels):
    """
        CHW to HWC
    """
    o_ranges = {'uint8':[0,255], 'int16': [-1024,3071]} ##TODO #[-32768, 32767]}
    assert o_dtype in o_ranges
    o_range = o_ranges[o_dtype]
    i_range = [image.min(), image.max()]

    #print("Raw image", image.min(), image.max())

    if o_dtype == 'uint8':
        image = percentile(image, p=(0, 99), o_range=o_range) #CHW
    else:
        image = dynamic_range(image, i_range=i_range, o_range=o_range, o_dtype=o_dtype) # CHW
        #print("After Dynamic range adjustment", image.min(), image.max())

    if o_channels == 3:
        image = np.tile(image, (3,1,1)) # for CHW

    return image # CHW
    #return np.transpose(image, (1, 2, 0)) # HWC for np.ndarray


class DICOMTransform(object):
    import cv2

    def __init__(self, resolution, output_dtype, pad, resize, norm):
        self.resolution = resolution
        self.output_dtype = output_dtype
        assert self.output_dtype in ['uint8', 'int16']
        self.pad = pad
        self.resize = resize
        self.norm = norm

    def __call__(self, image):
        """
           Input image should have HxWxC order
        """
        image = np.transpose(image, (1, 2, 0)) # HWC for np.ndarray
        h, w, c = image.shape
        assert c in [1, 3]
        #TODO: norm
        
        if self.pad:
            image = self._padding(image)
        if self.resize:
            image = self._resize(image) 
        if self.norm:
            image = self._normalize(image)
        ## stylegan2 input should be CHW
        image = np.transpose(image, (2, 0, 1)) # HWC to CHW
        return image


    def _padding(self, image):
        if self.output_dtype == 'uint8':
            return padding(image, const_value=0)
        elif self.output_dtype == 'int16':
            return padding(image, const_value=-1024)

    def _resize(self, image):
        return resize(image, (self.resolution, self.resolution)) #HWC


    def _normalize(self, image):
        o_dtype = self.output_dtype
        o_ranges = {'uint8':[0,255], 'int16': [-1024,3071]} #TODO:
        return normalize(image, o_ranges[o_dtype], mean=0.5, std=0.5)



def _data_transforms_dicom(resolution, o_dtype, pad=True, resize=True, norm=False):
    return DICOMTransform(resolution, o_dtype, pad, resize, norm)


"""
def _data_transforms_dicom_xray(resolution, norm=True):
    import torchvision.transforms as transforms

    #TODO: 3 channel transform
    return transforms.Compose([

        ToPILImage(), #CHW to HWC
        SquarePad(), # HWC
        ToTensor(), # HWC to CHW
        transforms.Resize(resolution), #CHW to HWC
        Normalize((0.5,), (0.5,))
    ])
"""
