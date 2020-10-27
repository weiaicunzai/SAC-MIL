import numbers
import random
from collections.abc import Iterable
from PIL import Image

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F

#class RandomHorizontalFlip:
#    """Horizontally flip the given opencv image with given probability p.
#    and does the same to mask
#    Args:
#        p: probability of the image being flipped
#    """
#    def __init__(self, p=0.5):
#        self.p = p
#
#    def __call__(self, img, mask):
#        """
#        Args:
#            the image to be flipped
#        Returns:
#            flipped image
#        """
#        if random.random() < self.p:
#            img = cv2.flip(img, 1)
#            mask = cv2.flip(mask, 1)
#
#        return img, mask
#
#class ToPILImage(object):
#    """Convert a tensor or an ndarray to PIL Image.
#
#    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
#    H x W x C to a PIL Image while preserving the value range.
#
#    Args:
#        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
#            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
#             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
#             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
#             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
#             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
#               ``short``).
#
#    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
#    """
#    def __init__(self, mode=None):
#        self.mode = mode
#
#    def __call__(self, pic, mask):
#        """
#        Args:
#            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
#
#        Returns:
#            PIL Image: Image converted to PIL Image.
#            PIL Image: Mask converted to PIL Image.
#
#        """
#        F.to_pil_image(pic, self.mode)
#        F.to_pil_image(mask, mode='L')
#        return pic, mask
#
#
#    def __repr__(self):
#        format_string = self.__class__.__name__ + '('
#        if self.mode is not None:
#            format_string += 'mode={0}'.format(self.mode)
#        format_string += ')'
#        return format_string
#
#class Compose:
#    """Composes several transforms together.
#    Args:
#        transforms(list of 'Transform' object): list of transforms to compose
#    """
#
#    def __init__(self, transforms):
#        self.transforms = transforms
#
#    def __call__(self, img, mask):
#
#        for trans in self.transforms:
#            print(trans)
#            img, mask = trans(img, mask)
#
#        return img, mask
#
#    def __repr__(self):
#        format_string = self.__class__.__name__ + '('
#        for t in self.transforms:
#            format_string += '\n'
#            format_string += '    {0}'.format(t)
#        format_string += '\n)'
#        return format_string
#
##class ToPILImage(torchvision.transforms.ToPILImage):
##    """a wrapper for torchvision topilimage"""
##    def __init__(self, mode=None):
##        super().__init__(mode=mode)
##
##class RandomRotation(torchvision.transforms.RandomRotation):
##    """a wrapper for torchvison random rotation"""
##    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
##        super().__init__(degrees, resample=resample, expand=expand, center=center, fill=fill)
#class RandomRotation:
#    """Rotate the image by angle
#    Args:
#        angle: rotated angle
#        value: value used for filling the empty pixel after rotating,
#               should use ignore class index
#    """
#
#    def __init__(self, angle=10, value=0):
#
#        if not (isinstance(angle, numbers.Number) and angle > 0):
#            raise ValueError('angle must be a positive number.')
#
#        self.angle = angle
#        self.value = value
#
#    def __call__(self, image, mask):
#
#        angle = random.uniform(-self.angle, self.angle)
#        image_center = tuple(np.array(image.shape[1::-1]) / 2)
#        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#        image = cv2.warpAffine(
#            image, rot_mat, image.shape[1::-1])
#        mask = cv2.warpAffine(
#            mask, rot_mat, mask.shape[1::-1],
#            flags=cv2.INTER_NEAREST,
#            borderMode=cv2.BORDER_CONSTANT,
#            borderValue=self.value
#        )
#
#        return image, mask
#
#class ColorJitter:
#
#    """Randomly change the brightness, contrast and saturation of an image
#    Args:
#        brightness: (float or tuple of float(min, max)): how much to jitter
#            brightness, brightness_factor is choosen uniformly from[max(0, 1-brightness),
#            1 + brightness] or the given [min, max], Should be non negative numbe
#        contrast: same as brightness
#        saturation: same as birghtness
#        hue: same as brightness
#    """
#
#    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
#        self.brightness = self._check_input(brightness)
#        self.contrast = self._check_input(contrast)
#        self.saturation = self._check_input(saturation)
#        self.hue = self._check_input(hue)
#
#    def _check_input(self, value):
#
#        if isinstance(value, numbers.Number):
#            assert value >= 0, 'value should be non negative'
#            value = [max(0, 1 - value), 1 + value]
#
#        elif isinstance(value, (list, tuple)):
#            assert len(value) == 2, 'brightness should be a tuple/list with 2 elements'
#            assert 0 <= value[0] <= value[1], 'max should be larger than or equal to min,\
#            and both larger than 0'
#
#        else:
#            raise TypeError('need to pass int, float, list or tuple, instead got{}'.format(type(value).__name__))
#
#        return value
#
#    def __call__(self, img, mask):
#        """
#        Args:
#            img to be jittered
#        Returns:
#            jittered img
#        """
#
#        img_dtype = img.dtype
#        h_factor = random.uniform(*self.hue)
#        b_factor = random.uniform(*self.brightness)
#        s_factor = random.uniform(*self.saturation)
#        c_factor = random.uniform(*self.contrast)
#
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#        img = img.astype('float32')
#
#        #h
#        img[:, :, 0] *= h_factor
#        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)
#
#        #s
#        img[:, :, 1] *= s_factor
#        img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)
#
#        #v
#        img[:, :, 2] *= b_factor
#        img[:, :, 2] = np.clip(img[:, :, 2], 0, 255)
#
#        img = img.astype(img_dtype)
#        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
#
#        #c
#        img = img * c_factor
#        img = img.astype(img_dtype)
#        img = np.clip(img, 0, 255)
#
#        return img, mask
#
#class Resize:
#    """Resize an image and an mask to given size
#    Args:
#        size: expected output size of each edge, can be int or iterable with (w, h)
#    """
#
#    def __init__(self, size):
#
#        if isinstance(size, int):
#            self.size = (size, size)
#        elif isinstance(size, Iterable) and len(size) == 2:
#            self.size = size
#        else:
#            raise TypeError('size should be iterable with size 2 or int')
#
#    def __call__(self, img, mask):
#
#        resized_img = cv2.resize(img, self.size)
#        resized_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
#
#        return resized_img, resized_mask
#
#
#
#class RandomTransforms(object):
#    """Base class for a list of transformations with randomness
#
#    Args:
#        transforms (list or tuple): list of transformations
#    """
#
#    def __init__(self, transforms):
#        assert isinstance(transforms, (list, tuple))
#        self.transforms = transforms
#
#    def __call__(self, *args, **kwargs):
#        raise NotImplementedError()
#
#    def __repr__(self):
#        format_string = self.__class__.__name__ + '('
#        for t in self.transforms:
#            format_string += '\n'
#            format_string += '    {0}'.format(t)
#        format_string += '\n)'
#        return format_string
#
##class Resize:
##    """Resize an image and an mask to given size
##    Args:
##        size: expected output size of each edge, can be int or iterable with (w, h)
##    """
##
##    def __init__(self, size):
##
##        if isinstance(size, size):
##            self.size = (size, size)
##        elif isinstance(size, Iterable) and len(size) == 2:
##            self.size = size
##        else:
##            raise TypeError('size should be iterable with size 2 or int')
##
##    def __call__(self, img, mask):
##
##        resized_img = cv2.resize(img, self.size)
##        resized_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
##
##        return resized_img, resized_mask
##class Resize(torchvision.transforms.Resize):
##    """a wrapper for torchvison random resize"""
##    def __init__(self, size, interpolation=2):
##        super().__init__(size, interpolation=interpolation)
#
#class GaussianBlur():
#    def __init__(self, p=0.2, sigma=(0.5, 1.5)):
#        """apply gaussian blur to an PIL image
#        Args:
#            p: probility for apply gaussian blur
#            sigma: sigma value interval, uniform distribution
#        """
#        self.p = p
#        self.simga = sigma
#
#    #def __call__(self, img, kas):
#
##class
##help(Resize)
##resize = Resize(33, 0)
##print(resize)
#
#
## roation = RandomRotation(13)
## print(roation)



class Compose:
    """Composes several transforms together.
    Args:
        transforms(list of 'Transform' object): list of transforms to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):

        for trans in self.transforms:
            print(trans)
            img, mask = trans(img, mask)

        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Resize:
    """Resize an image and an mask to given size
    Args:
        size: expected output size of each edge, can be int or iterable with (w, h)
    """

    def __init__(self, size):

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, Iterable) and len(size) == 2:
            self.size = size
        else:
            raise TypeError('size should be iterable with size 2 or int')

    def __call__(self, img, mask):

        resized_img = cv2.resize(img, self.size)
        resized_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        return resized_img, resized_mask

class RandomScale:
    """Randomly scaling an image (from 0.5 to 2.0]), then scale back to original shape,
    the output image and mask shape will be the same as the input image and mask shape.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        value: value to fill the mask when resizing,
               should use ignore class index
    """

    def __init__(self, p=0.25, scale=(0.5, 1.0), value=0):

        if not isinstance(scale, Iterable) and len(scale) == 2:
            raise TypeError('scale should be iterable with size 2 or int')

        self.value = value
        self.scale = scale
        self.p = p

    def __call__(self, img, mask):

        if random.random() >= self.p:
            return img, mask

        oh, ow = img.shape[:2]

        # scale image
        scale = random.uniform(*self.scale)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (oh, ow))

        return img, mask

class GaussianNoise:
    def __init__(self, sigma=(0, 0.1)):
        self.sigma = sigma

    def __call__(self, img, mask):
        dtype = img.dtype
        sigma = random.uniform(*self.sigma)
        noise = np.random.normal(0, sigma, img.shape)

        img = img + noise
        img = img.astype(dtype)
        img = np.clip(img, 0, 255)

        return img, mask


class RandomRotation:
    """Rotate the image by angle
    Args:
        angle: rotated angle
        value: value used for filling the empty pixel after rotating,
               should use ignore class index
    """

    def __init__(self, angle=10, value=0):

        if not (isinstance(angle, numbers.Number) and angle > 0):
            raise ValueError('angle must be a positive number.')

        self.angle = angle
        self.value = value

    def __call__(self, image, mask):

        angle = random.uniform(-self.angle, self.angle)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1])
        mask = cv2.warpAffine(
            mask, rot_mat, mask.shape[1::-1],
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.value
        )

        return image, mask

class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.
    and does the same to mask
    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        return img, mask

class GaussianBlur:

    """Blur an image using gaussian blurring.
    Args:
       sigma: Standard deviation of the gaussian kernel.
       Values in the range ``0.0`` (no blur) to ``3.0`` (strong blur) are
       common. Kernel size will automatically be derived from sigma
       p: probability of applying gaussian blur to image
       https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmenters/blur.html#GaussianBlur
    """

    def __init__(self, sigma=(0.0, 3.0)):

        if not isinstance(sigma, Iterable) and len(sigma) == 2:
            raise TypeError('sigma should be iterable with length 2')

        if not sigma[1] >= sigma[0] >= 0:
            raise ValueError(
                'sigma shoule be an iterval of nonegative real number')

        self.sigma = sigma

    def __call__(self, img, mask):

        sigma = random.uniform(*self.sigma)
        k_size = self._compute_gaussian_blur_ksize(sigma)
        img = cv2.GaussianBlur(img, (k_size, k_size),
                               sigmaX=sigma, sigmaY=sigma)

        return img, mask

    @staticmethod
    def _compute_gaussian_blur_ksize(sigma):
        if sigma < 3.0:
            ksize = 3.3 * sigma  # 99% of weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # 97% of weight
        else:
            ksize = 2.6 * sigma  # 95% of weight

        ksize = int(max(ksize, 3))

        # kernel size needs to be an odd number
        if not ksize % 2:
            ksize += 1

        return ksize

class ColorJitter:

    """Randomly change the brightness, contrast and saturation of an image
    Args:
        brightness: (float or tuple of float(min, max)): how much to jitter
            brightness, brightness_factor is choosen uniformly from[max(0, 1-brightness),
            1 + brightness] or the given [min, max], Should be non negative numbe
        contrast: same as brightness
        saturation: same as birghtness
        hue: same as brightness
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, mask):
        """
        Args:
            img to be jittered
        Returns:
            jittered img
        """

        trans = torchvision.transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
        img = Image.fromarray(img)
        img = trans(img)
        img = np.array(img)

        return img, mask

class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch
    float tensor (c, h, w) ranged from 0 to 1, and convert mask to torch tensor
    """

    def __call__(self, img, mask):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]
        Returns:
            a pytorch tensor range from [0., 1.0]
        """
        #convert format H W C to C H W

        image = F.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask

class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    and does nothing to mask tensor
    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):

        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, mask

class RandomChoice():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        trans = random.choice(self.transforms)
        return trans(img, mask)

class RandomApply():
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask

        for t in self.transforms:
            img, mask = t(img, mask)

        return img, mask
