import os
import sys
import numpy as np
import SimpleITK as sitk

sitk_type = {'uint8': sitk.sitkUInt8, 
            'uint16': sitk.sitkUInt16,
            'int16': sitk.sitkInt16,
            'int32': sitk.sitkInt32}

#------------------------------------------------------------------------------------#

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#--------------------------------------------------------------------------------------#

def dcm_to_array(path:str, dtype='int16'):
    """
        path: image file path
        dtype: data type to cast, default:int16
    """
    dcm = sitk.GetArrayFromImage(sitk.ReadImage(path))

    ##TODO:
    if dcm.dtype in ['float32', 'float64']: #Temp casting
        return dcm.astype('int16')

    return dcm.astype(dtype)

def array_to_pil(img):
    from PIL import Image

    if img.dtype == 'int32':
        mode = 'I'
    elif img.dtype == 'uint16':
        mode = 'I;16'
    elif img.dtype == 'int8':
        if len(img.shape) == 2:
            mode = 'L'
        elif len(img.shape) == 3: #TODO
            mode = 'RGB'
    else:
        error("Image type should be either int32, uint16 or int8")
    return Image.fromarray(img, mode=mode)

#-------------------------------------------------------------------------------------#

def clip(img, center=1024, width=4096):
    """ 
    Clip DICOM range 
    img: array
    """
    img = np.array(img)

    lower = center - width // 2
    upper = center + width // 2
    img[img < lower] = lower
    img[img > upper] = upper
    return img


def normalize(img, d_range, mean=0.5, std=0.5):
    """ 
    Normalize to output range. ex) [0, 1] or [-1, 1]
    d_range: dynamic range of img
    o_range: dynamic range of the output
    """
    min, max = d_range
    img = (img - min) / (max - min)
    img[img <= 0] = 0
    img[img >= 1] = 1
    
    img = (img - mean) / std
    return img


def dynamic_range(img, i_range=[0, 65535], o_range=[-1024, 3071], o_dtype='int16'):
    """
    Change dynamic range of input data
    ex) [0, 65535] -> [-1024, 3071]
    dtype: output data type
    """
    lower, upper = i_range
    o_width = o_range[1] - o_range[0]

    img = np.asarray(img, dtype=np.float32)
    img = (img - lower) * (o_width / (upper - lower)) + o_range[0]
    img = np.rint(img).clip(*o_range).astype(o_dtype)
    return img


#------------------------------------Windowing----------------------------------------#
def array_to_sitk_image(img):
    return sitk.GetImageFromArray(img)

def sitk_image_to_array(img_meta):
    return sitk.GetArrayFromImage(img_meta)

def percentile(img, p=(0, 99), o_range=[0, 255]):
    dtype = img.dtype
    min = np.percentile(img, p[0])
    max = np.percentile(img, p[1])
    
    img[img <= min] = min
    img[img >= max] = max

    img = (img - min) / (max - min)
    img *= (o_range[1] - o_range[0])
    img -= o_range[0]
    return np.array(img).astype(dtype)


def padding(img, const_value=0):
    """
       Image shape should be HxWxC
    """

    assert img.shape[-1] in [1, 3], print("Input shape should be HxWxC")
    h, w, c = img.shape

    max_wh = np.max([w, h])
    hp = (max_wh - w) // 2
    vp = (max_wh - h) // 2

    if max_wh == w:
        img = np.pad(img, ((vp, vp), (0, 0), (0, 0)), 'constant', constant_values=const_value)
    else:
        img = np.pad(img, ((0, 0), (hp, hp), (0, 0)), 'constant', constant_values=const_value)
    return img


def resize(img, shape, interpolation=None):
    import cv2
    #TODO: interpolation setting
    interpolation = cv2.INTER_NEAREST #cv2.INTER_LANCZOS4
    assert img.shape[-1] in [1, 3], print("Input shape should be HxWxC")

    img = cv2.resize(img, shape, interpolation=interpolation)

    if len(img.shape) == 2:
        return img[:, :, np.newaxis]
    else:
        return img


def window(img_meta, w_range, o_range=[0, 255], o_dtype='uint8'):
    """
    Apply windowing
    w_range: windowing range to apply
    o_range: dynamic range of output 
    o_dtype: output data type
    """
    img_meta = sitk.IntensityWindowing(img_meta,
                                      windowMinimum=w_range[0],
                                      windowMaximum=w_range[1],
                                     outputMinimum=o_range[0],
                                    outputMaximum=o_range[1])
    return sitk.Cast(img_meta, sitk_type[o_dtype])

def window_and_stack(img_meta, 
        window1=(0,80), 
        window2=(-20,180), 
        window3=(-800, 2000),
        o_dtype='uint8'):

    """
    Stack 3 different windowing images.
    Composition may be changable. (stack order)
    """
    img1 = window(img_meta, window1, o_dtype=o_dtype)
    img2 = window(img_meta, window2, o_dtype=o_dtype)
    img3 = window(img_meta, window3, o_dtype=o_dtype)

    return sitk.Compose(img1, img2, img3) ## composition might be changed

def apply_window(img, w_range, o_range=[0, 255], o_dtype='uint8'):
    img_meta = sitk.GetImageFromArray(img)
    return sitk.GetArrayFromImage(window(img_meta, w_range))









