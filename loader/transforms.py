import random
import math
import numpy as np
import torch
import cv2

from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 65535] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self):
        self.norm_value = 2**16-1

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        default_float_dtype = torch.get_default_dtype()

        ## TODO: dealing with numpy input
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.to(dtype=default_float_dtype).div(self.norm_value)
            else:
                return img

        if accimage is not None and isinstance(pic, accimage.Image):

            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic).to(dtype=default_float_dtype)
    
        # handle PIL Image
        ## I;16 PIL.Image is stored in unsigned 16bit type [0-65535],
        ## But torch.from_numpy() only support int16
        ## Changed pic type of I;16 to np.int32 for save conversion
        #mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
        mode_to_nptype = {'I': np.int32, 'I;16': np.int32, 'F':np.float32}
        pic_mode = pic.mode
        pic_size = pic.size
        pic_bands = pic.getbands()
        np_mode  = mode_to_nptype.get(pic_mode, np.uint16)

        img = np.array(pic, np_mode, copy=True)
        img = torch.from_numpy(img)
        
        if pic_mode == '1':
            img = self.norm_value * img
        img = img.view(pic_size[1], pic_size[0], len(pic_bands))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype).div(self.norm_value)
        else:
            #return img
            return img.to(dtype=default_float_dtype).div(self.norm_value)
            
            
class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image. This transform does not support torchscript.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
            - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
            - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
            ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """

    def __init__(self, mode=None):
        self.mode = mode

        
    def to_pil_image(self, pic, mode=None):
        if not(isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

        elif isinstance(pic, torch.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

            # check number of channels
            if pic.shape[-3] > 4:
                raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-3]))

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

            # check number of channels
            if pic.shape[-1] > 4:
                raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-1]))

        npimg = pic
        if isinstance(pic, torch.Tensor):
            if pic.is_floating_point() and mode != 'F':
                pic = pic.mul(255).byte()
            npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                            'not {}'.format(type(npimg)))

        
        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            ## uint16 is not supported. changed to np.int32 for save conversion
            elif npimg.dtype == np.uint16:
                expected_mode = 'I;16'
            if mode is not None and mode != expected_mode:
                raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                                 .format(mode, np.dtype, expected_mode))
            mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if mode is not None and mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if mode is not None and mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if mode is not None and mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
            if mode is None and npimg.dtype == np.uint8:
                mode = 'RGB'

        if mode is None:
            raise TypeError('Input type {} is not supported'.format(npimg.dtype))

        
        return Image.fromarray(npimg, mode=mode)

        
        
    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        
        return self.to_pil_image(pic, self.mode)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += f"mode={self.mode}"
        format_string += ")"
        return format_string        


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        
        if not self.inplace:
            tensor = tensor.clone()
        
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
    
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.sub_(mean).div_(std)
        return tensor



class SquarePad(object):

    def pad(self, img, padding, fill=0, padding_mode="constant"):
        import torchvision.transforms.functional_pil as F_pil
        import torchvision.transforms.functional_tensor as F_t
        if not isinstance(img, torch.Tensor):
            return F_pil.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

        return F_t.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)


        return self.pad(image, padding, 0, 'constant')


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)
    

