import numpy as np
from PIL import Image

import utils.visualize as visualizer
import augmentation.transform_function as transform_func


class AugmentName:
    IDENTITY = transform_func.identity.__name__
    TRANSLATE_X = transform_func.translate_x.__name__
    TRANSLATE_Y = transform_func.translate_y.__name__
    SCALE_WIDTH = transform_func.scale_width.__name__
    SCALE_HEIGHT = transform_func.scale_height.__name__
    ROTATE = transform_func.rotate.__name__
    SHEAR_X = transform_func.shear_x.__name__
    SHEAR_Y = transform_func.shear_y.__name__
    COLOR = transform_func.color.__name__
    POSTERIZE = transform_func.posterize.__name__
    SOLARIZE = transform_func.solarize.__name__
    CONTRAST = transform_func.contrast.__name__
    SHARPNESS = transform_func.sharpness.__name__
    BRIGHTNESS = transform_func.brightness.__name__
    AUTOCONSTRAST = transform_func.autocontrast.__name__
    INVERT = transform_func.invert.__name__

    @staticmethod
    def get_names():
        return [
            AugmentName.IDENTITY,
            AugmentName.TRANSLATE_X,
            AugmentName.TRANSLATE_Y,
            AugmentName.SCALE_WIDTH,
            AugmentName.SCALE_HEIGHT,
            AugmentName.ROTATE,
            AugmentName.SHEAR_X,
            AugmentName.SHEAR_Y,
            AugmentName.COLOR,
            AugmentName.POSTERIZE,
            AugmentName.SOLARIZE,
            AugmentName.CONTRAST,
            AugmentName.SHARPNESS,
            AugmentName.BRIGHTNESS,
            AugmentName.AUTOCONSTRAST,
            AugmentName.INVERT,
        ]


class DERandAugment:
    TRANSFORMS = {name: getattr(transform_func, name) for name in AugmentName.get_names()}

    RANGES = {
        AugmentName.IDENTITY: np.linspace(0, 0, 10),
        AugmentName.TRANSLATE_X: np.linspace(0, 0.2, 10),
        AugmentName.TRANSLATE_Y: np.linspace(0, 0.2, 10),
        AugmentName.SCALE_WIDTH: np.linspace(0, 0.1, 10),
        AugmentName.SCALE_HEIGHT: np.linspace(0, 0.1, 10),
        AugmentName.ROTATE: np.linspace(0, 60, 10),
        AugmentName.SHEAR_X: np.linspace(0, 0.3, 10),
        AugmentName.SHEAR_Y: np.linspace(0, 0.3, 10),
        AugmentName.COLOR: np.linspace(0.0, 0.9, 10),
        AugmentName.POSTERIZE: np.round(np.linspace(8, 4, 10), 0).astype(np.int),
        AugmentName.SOLARIZE: np.linspace(256, 231, 10),
        AugmentName.CONTRAST: np.linspace(0.0, 0.5, 10),
        AugmentName.SHARPNESS: np.linspace(0.0, 0.9, 10),
        AugmentName.BRIGHTNESS: np.linspace(0.0, 0.3, 10),
        AugmentName.AUTOCONSTRAST: np.linspace(0, 0, 10),
        AugmentName.INVERT: np.linspace(0, 0, 10),
    }

    def __init__(self, n_apply_transform=5, magnitude=6):
        self.n_apply_transform = n_apply_transform
        self.magnitude = magnitude

        return

    def __call__(self, images, annotations=[]):
        assert len(images) == len(annotations) or len(annotations) == 0, "Number of anotation must be zero or equal to number of image"

        result = []
        for i in range(len(images)):
            if len(annotations) > 0:
                annotation = annotations[i]

            image, annotation = self.transform(
                image=images[i],
                annotations=annotation
            )

            result.append({"image": image, "annotations": annotation})

        return result

    def transform(self, image, annotations):
        ops = np.random.choice(list(DERandAugment.TRANSFORMS.values()), self.n_apply_transform)
        ops = [ops[i](image, DERandAugment.RANGES[ops[i].__name__][np.random.randint(low=0, high=self.magnitude)]) for i in range(0, len(ops))]

        for i in range(len(ops)):
            image, annotations = ops[i](image, annotations)

        return image, annotations


# image = Image.open("test.png")
# print(image.size)
# annotation = {
#     "bbox": [100, 100, 300, 300],
#     "label": "Funny"
# }
#
# annotation2 = {
#     "bbox": [100, 200, 300, 300],
#     "label": "Funny"
# }
#
# visualizer.show_image(
#         image=image,
#         annotations=[annotation]
#     )
#
# aug = DERandAugment(n_apply_transform=7, magnitude=5)
# for i in range(20):
#     image = aug(images=[image], annotations=[[annotation]])
#     print(annotation)
#     visualizer.show_image(
#         image=result[0]["image"],
#         annotations=result[0]["annotations"],
#     )


