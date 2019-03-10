import math
from scipy import ndimage
import numpy as np
import cv2

# see http://opensourc.es/blog/tensorflow-mnist
def normalize_image(img,pixels=64):
  img = delete_white(img)
  img = resize_to(img, pixels=pixels)
  img = zero_pad(img, pixels=pixels)
  shiftx,shifty = getBestShift(img)
  shifted = shift(img,shiftx,shifty)
  return 1-img

# binarize
def delete_white(img):
  (thresh, img) = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  while np.sum(img[0]) == 0:
        img = img[1:]
  while np.sum(img[:,0]) == 0:
      img = np.delete(img,0,1)
  while np.sum(img[-1]) == 0:
      img = img[:-1]
  while np.sum(img[:,-1]) == 0:
    img = np.delete(img,-1,1)
  return img

# resize to pixels
def resize_to(img, pixels=64): 
  rows,cols = img.shape
  if rows > cols:
    factor = 1.0*pixels/rows
    rows = pixels
    cols = int(round(cols*factor))
    img = cv2.resize(img, (cols,rows))
  else:
    factor = 1.0*pixels/cols
    cols = pixels
    rows = int(round(rows*factor))
    img = cv2.resize(img, (cols, rows))
  return img

def zero_pad(img, pixels=64):
  rows,cols = img.shape
  colsPadding = (int(math.ceil((pixels-cols)/2.0)),int(math.floor((pixels-cols)/2.0)))
  rowsPadding = (int(math.ceil((pixels-rows)/2.0)),int(math.floor((pixels-rows)/2.0)))
  img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
  return img

def getBestShift(img):
  cy,cx = ndimage.measurements.center_of_mass(img)

  rows,cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted  