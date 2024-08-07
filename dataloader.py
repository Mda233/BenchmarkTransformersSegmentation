import os
import torch
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)



def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_transform_segmentation():
  AUGMENTATIONS_TRAIN = Compose([
    ShiftScaleRotate(rotate_limit=10),
    RandomBrightnessContrast(),
    ToFloat(max_value=1)
    ],p=1)

  return AUGMENTATIONS_TRAIN




class ChestXray14Dataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpertDataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, annotation_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):

        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):

        return len(self.img_list)

# ---------------------------------------------Downstream RSNA Pneumonia------------------------------------------
class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, augment, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          self.img_label.append(int(lineItems[-1]))

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = np.zeros(3)
    imageLabel[self.img_label[index]] = 1
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Segmentation------------------------------------------

class Montgomery(Dataset):

    def __init__(self, images_path, file_path, augment, image_size=(224,224), anno_percent=100, normalization=None):
        self.augmentation = augment

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.normalization = normalization

        with open(file_path, "r") as fileDescriptor:
            line = fileDescriptor.readline().strip()
            while line:
                self.img_list.append(os.path.join(images_path + "/CXR_png", line))
                self.img_label.append(
                    (os.path.join(images_path+"/ManualMask/leftMask", line),(os.path.join(images_path+"/ManualMask/rightMask", line)))
                      )
                line = fileDescriptor.readline().strip()
        
        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        image = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE), self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE), self.image_size, interpolation=cv2.INTER_AREA)
        
        mask = leftMaskData + rightMaskData
        mask[mask > 0] = 255
  
        if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image=augmented['image']
                mask=augmented['mask']
                image=np.array(image) / 255.
                mask=np.array(mask) / 255.
        else:
            image = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std

        # mask = np.array(mask) / 255.
        # image = np.array(image) / 255.
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # image = (image-mean)/std

        image = image.transpose(2, 0, 1).astype('float32')
        mask = np.expand_dims(mask,axis=0).astype('uint8')
        return image, mask


class JSRTLung(Dataset):

    def __init__(self, images_path, file_path, augment, image_size=(224,224), anno_percent=100, normalization=None):
        self.augmentation = augment

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.normalization = normalization

        with open(file_path, "r") as fileDescriptor:
            line = fileDescriptor.readline().strip()
            while line:
                self.img_list.append(os.path.join(images_path + "/images", line+".IMG.png"))
                self.img_label.append(
                    (os.path.join(images_path+"/masks/left_lung_png", line+".png"),(os.path.join(images_path+"/masks/right_lung_png", line+".png")))
                      )
                line = fileDescriptor.readline().strip()
        
        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        image = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE), self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE), self.image_size, interpolation=cv2.INTER_AREA)
        
        mask = leftMaskData + rightMaskData
        mask[mask > 0] = 255
  
        if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image=augmented['image']
                mask=augmented['mask']
                image=np.array(image) / 255.
                mask=np.array(mask) / 255.
        else:
            image = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std

        image = image.transpose(2, 0, 1).astype('float32')
        mask = np.expand_dims(mask,axis=0).astype('uint8')
        return image, mask

class JSRTClavicle(Dataset):

    def __init__(self, images_path, file_path, augment, image_size=(224,224), few_shot=0, normalization=None):
        self.augmentation = augment

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.normalization = normalization

        with open(file_path, "r") as fileDescriptor:
            line = fileDescriptor.readline().strip()
            while line:
                self.img_list.append(os.path.join(images_path + "/images", line+".IMG.png"))
                self.img_label.append(
                    (os.path.join(images_path+"/masks/left_clavicle_png/", line+".png"),(os.path.join(images_path+"/masks/right_clavicle_png/", line+".png")))
                      )
                line = fileDescriptor.readline().strip()
        
        indexes = np.arange(len(self.img_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        image = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE), self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE), self.image_size, interpolation=cv2.INTER_AREA)
        
        mask = leftMaskData + rightMaskData
        mask[mask > 0] = 255
  
        if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image=augmented['image']
                mask=augmented['mask']
                image=np.array(image) / 255.
                mask=np.array(mask) / 255.
        else:
            image = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std

        image = image.transpose(2, 0, 1).astype('float32')
        mask = np.expand_dims(mask,axis=0).astype('uint8')
        return image, mask

class JSRTHeart(Dataset):

    def __init__(self, images_path, file_path, augment, image_size=(224,224), anno_percent=100, normalization=None):
        self.augmentation = augment

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.normalization = normalization

        with open(file_path, "r") as fileDescriptor:
            line = fileDescriptor.readline().strip()
            while line:
                self.img_list.append(os.path.join(images_path + "/images", line+".IMG.png"))
                self.img_label.append(os.path.join(images_path+"/masks/heart_png/", line+".png"))
                line = fileDescriptor.readline().strip()
        
        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        image = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        
        mask[mask > 0] = 255
  
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image=augmented['image']
            mask=augmented['mask']
            image=np.array(image) / 255.
            mask=np.array(mask) / 255.
        else:
            image = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std

        image = image.transpose(2, 0, 1).astype('float32')
        mask = np.expand_dims(mask,axis=0).astype('uint8')
        return image, mask
