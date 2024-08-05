import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance

from ..utils import sev_scheduler

mean_std_dict = {
  "cifar10"       : {
    "mean" : (0.4914, 0.4822, 0.4465),
    "std"  : (0.2470, 0.2435, 0.2616),
  },
    "cifar100"    : {
    "mean" : (0.5071, 0.4867, 0.4408),
    "std"  : (0.2675, 0.2565, 0.2761),
  },
  "tinyimagenet"  : {
    "mean" : (0.4803, 0.4481, 0.3976),
    "std"  : (0.2764, 0.2689, 0.2816),
  },
  "svhn"          : {
    "mean" : (0.4377, 0.4438, 0.4728),
    "std"  : (0.1980, 0.2010, 0.1970),
  },
  "chaoyang"      : {
    "mean" : (0.6470, 0.5523, 0.6694),
    "std"  : (0.2113, 0.2503, 0.1738),
  },
}


def get_augmix_dataset(train_set, cfg):
  train_set = AugMixDataset(train_set, cfg)
  return train_set


class AugMixDataset(Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, cfg):
    self.dataset = dataset
    if cfg.DATASET.IN_NORMALIZATION:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet normalization
    else:
        mean, std = mean_std_dict[cfg.DATASET.TYPE.TRAIN]["mean"], mean_std_dict[cfg.DATASET.TYPE.TRAIN]["std"]
    self.preprocess = transforms.Compose(
      [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
      ]
    )
    self.cfg = cfg
    self.severity_student = sev_scheduler(cfg, epoch=0)
    self.severity_teacher = sev_scheduler(cfg, epoch=0)

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.cfg.AUGMIX.JSD:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.cfg, self.severity_student),
                  aug(x, self.preprocess, self.cfg, self.severity_student))
      return im_tuple, y
    else:
      return aug(x, self.preprocess, self.cfg, self.severity_student), y

  def __len__(self):
    return len(self.dataset)


def aug(image, preprocess, cfg, severity):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations
  if cfg.AUGMIX.ALL_OPS:
    aug_list = augmentations_all

  ws = np.float32(np.random.dirichlet([1] * cfg.AUGMIX.MIXTURE_WIDTH))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(cfg.AUGMIX.MIXTURE_WIDTH):
    image_aug = image.copy()
    depth = cfg.AUGMIX.MIXTURE_DEPTH if cfg.AUGMIX.MIXTURE_DEPTH > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


# ImageNet code should change this value
IMAGE_SIZE = 32


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, level):
  if level > 0:
    return ImageOps.autocontrast(pil_img)
  else:
    return pil_img


def equalize(pil_img, level):
  if level > 0:
    return ImageOps.equalize(pil_img)
  else:
    return pil_img


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), pil_img.size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), pil_img.size[0] / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]