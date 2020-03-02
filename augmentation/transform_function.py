import augmentation.def_function as def_function
import numpy as np
from PIL import ImageOps, ImageEnhance


@def_function.affine_transform
def identity(image, magnitude, sign=None):
    transform_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return transform_matrix


@def_function.affine_transform
def translate_x(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    transform_matrix = np.array([
        [1, 0, sign * magnitude * image.size[0]],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return transform_matrix


@def_function.affine_transform
def translate_y(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    transform_matrix = np.array([
        [1, 0, 0],
        [0, 1, sign * magnitude * image.size[1]],
        [0, 0, 1]
    ])

    return transform_matrix


@def_function.affine_transform
def scale_width(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    transform_matrix = np.array([
        [(1 + sign*magnitude), 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return transform_matrix


@def_function.affine_transform
def scale_height(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    transform_matrix = np.array([
        [1, 0, 0],
        [0, (1 + sign*magnitude), 0],
        [0, 0, 1]
    ])

    return transform_matrix


@def_function.affine_transform
def rotate(image, magnitude, sign=None):
    angle = np.deg2rad(magnitude)
    move_origin_to_origin = np.array([
        [1, 0, -0.5 * image.size[0]],
        [0, 1, -0.5 * image.size[1]],
        [0, 0, 1],
    ])

    rotate_matrix = np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    move_origin_to_center = np.array([
        [1, 0, 0.5 * image.size[0]],
        [0, 1, 0.5 * image.size[1]],
        [0, 0, 1],
    ])

    transform_matrix = np.dot(move_origin_to_center, np.dot(rotate_matrix, move_origin_to_origin))
    return transform_matrix


@def_function.affine_transform
def shear_x(image, magnitude, sign=None):
    transform_matrix = np.array([
        [1, magnitude, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return transform_matrix


@def_function.affine_transform
def shear_y(image, magnitude, sign=None):
    transform_matrix = np.array([
        [1, 0, 0],
        [magnitude, 1, 0],
        [0, 0, 1]
    ])

    return transform_matrix


@def_function.color_transform
def color(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    new_image = ImageEnhance.Color(image).enhance(1 + magnitude * sign)
    return new_image


@def_function.color_transform
def posterize(image, magnitude, sign=None):
    new_image = ImageOps.posterize(image, magnitude)
    return new_image


@def_function.color_transform
def solarize(image, magnitude, sign=None):
    new_image = ImageOps.solarize(image, magnitude)
    return new_image


@def_function.color_transform
def contrast(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    new_image = ImageEnhance.Contrast(image).enhance(1 + magnitude * sign)
    return new_image


@def_function.color_transform
def sharpness(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    new_image = ImageEnhance.Sharpness(image).enhance(1 + magnitude * sign)
    return new_image


@def_function.color_transform
def brightness(image, magnitude, sign=None):
    if sign is None:
        sign = np.random.choice([-1, 1])

    new_image = ImageEnhance.Brightness(image).enhance(1 + magnitude * sign)
    return new_image


@def_function.color_transform
def autocontrast(image, magnitude, sign=None):
    new_image = ImageOps.autocontrast(image)
    return new_image


@def_function.color_transform
def invert(image, magnitude, sign=None):
    new_image = ImageOps.invert(image)
    return new_image
