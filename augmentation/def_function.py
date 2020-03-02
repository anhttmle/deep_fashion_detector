import numpy as np
from config.enum import BoundingBox
from config.enum import ObjDetection
import copy
from PIL import Image


def affine_transform(func):
    def get_transform_func(image, magnitude, sign=None):
        def get_transform_matrix():
            transform_matrix = func(image, magnitude, sign)
            return transform_matrix

        def transform_with_matrix(transform_matrix):
            def transform_image(target_image, annotations=[]):
                #target_image = Image.Image()
                new_image = target_image.transform(
                    size=target_image.size,
                    method=Image.AFFINE,
                    data=np.ndarray.flatten(np.linalg.inv(transform_matrix)),
                    resample=Image.BILINEAR,
                    fillcolor=np.random.randint(low=0, high=255)
                )
                if annotations is None or len(annotations) <= 0:
                    return new_image, None
                else:
                    annotations = copy.deepcopy(annotations)
                    vectors = []
                    for _, annotation in enumerate(annotations):
                        bbox = annotation[ObjDetection.BBOX]
                        vectors.append([bbox[BoundingBox.TOP_LEFT_X], bbox[BoundingBox.TOP_LEFT_Y], 1])
                        vectors.append([bbox[BoundingBox.TOP_LEFT_X], bbox[BoundingBox.BOTTOM_RIGHT_Y], 1])
                        vectors.append([bbox[BoundingBox.BOTTOM_RIGHT_X], bbox[BoundingBox.BOTTOM_RIGHT_Y], 1])
                        vectors.append([bbox[BoundingBox.BOTTOM_RIGHT_X], bbox[BoundingBox.TOP_LEFT_Y], 1])


                    vectors = np.array(vectors).T
                    vectors = np.matmul(transform_matrix, vectors)

                    for index, annotation in enumerate(annotations):
                        bbox = annotation[ObjDetection.BBOX]
                        bbox[BoundingBox.TOP_LEFT_X] = np.min(vectors[0, (index*4):(index+1)*4])
                        bbox[BoundingBox.TOP_LEFT_Y] = np.min(vectors[1, (index*4):(index+1)*4])
                        bbox[BoundingBox.BOTTOM_RIGHT_X] = np.max(vectors[0, (index*4):(index+1)*4])
                        bbox[BoundingBox.BOTTOM_RIGHT_Y] = np.max(vectors[1, (index*4):(index+1)*4])

                    return new_image, annotations

            transform_image_func = transform_image
            transform_image_func.__name__ = func.__name__
            return transform_image

        return transform_with_matrix(transform_matrix=get_transform_matrix())

    f = get_transform_func
    f.__name__ = func.__name__

    return f


def color_transform(func):
    def get_transform_func(image, magnitude, sign=None):
        def transform_image(target_image, annotations=[]):
            new_image = func(target_image, magnitude, sign)

            return new_image, annotations

        transform_image_func = transform_image
        transform_image_func.__name__ = func.__name__
        return transform_image

    f = get_transform_func
    f.__name__ = func.__name__

    return f