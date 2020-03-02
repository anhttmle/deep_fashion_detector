from config.enum import ObjDetection, BoundingBox
import numpy as np


# image is PIL Image
# annotations is list of {ObjDetection.BBox: [x1 y1 x2 y2]}
def resize(image, annotations, target_size):
    origin_size = image.size
    image = image.resize(target_size)

    for annotation in annotations:
        annotation[ObjDetection.BBOX][BoundingBox.TOP_LEFT_X] *= target_size[ObjDetection.WIDTH]/origin_size[ObjDetection.WIDTH] * 1.0
        annotation[ObjDetection.BBOX][BoundingBox.TOP_LEFT_Y] *= target_size[ObjDetection.HEIGHT]/origin_size[ObjDetection.HEIGHT] * 1.0
        annotation[ObjDetection.BBOX][BoundingBox.BOTTOM_RIGHT_X] *= target_size[ObjDetection.WIDTH]/origin_size[ObjDetection.WIDTH] * 1.0
        annotation[ObjDetection.BBOX][BoundingBox.BOTTOM_RIGHT_Y] *= target_size[ObjDetection.HEIGHT]/origin_size[ObjDetection.HEIGHT] * 1.0

    return image, annotations


def scale_to_unit(image, annotations):
    origin_size = image.size
    for annotation in annotations:
        annotation[ObjDetection.BBOX][BoundingBox.TOP_LEFT_X] /= origin_size[ObjDetection.WIDTH]
        annotation[ObjDetection.BBOX][BoundingBox.TOP_LEFT_Y] /= origin_size[ObjDetection.HEIGHT]
        annotation[ObjDetection.BBOX][BoundingBox.BOTTOM_RIGHT_X] /= origin_size[ObjDetection.WIDTH]
        annotation[ObjDetection.BBOX][BoundingBox.BOTTOM_RIGHT_Y] /= origin_size[ObjDetection.HEIGHT]

    return np.array(image)/255., annotations
