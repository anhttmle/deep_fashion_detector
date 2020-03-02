import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import PIL.Image as Image

from config.enum import BoundingBox
from config.enum import ObjDetection


def show_image(image, annotations=None):
    if isinstance(image, Image.Image):
        image = np.array(image)

    assert isinstance(image, np.ndarray), "Image format is wrong"

    # Show image
    plt.imshow(image)

    ax = plt.gca()
    if annotations is None:
        return

    for annotation in annotations:
        bbox = annotation.get(ObjDetection.BBOX)
        if bbox is None:
            continue

            # Draw bounding box
        rect = Rectangle(
            xy=(bbox[BoundingBox.TOP_LEFT_X], bbox[BoundingBox.TOP_LEFT_Y]),
            width=bbox[BoundingBox.BOTTOM_RIGHT_X] - bbox[BoundingBox.TOP_LEFT_X],
            height=bbox[BoundingBox.BOTTOM_RIGHT_Y] - bbox[BoundingBox.TOP_LEFT_Y],
            fill=False,
            color='red')
        ax.add_patch(rect)

        label = annotation.get(ObjDetection.LABEL)
        if label is not None:
            # Write label
            label_text = "Label = {}".format(label)
            plt.text(
                x=bbox[BoundingBox.TOP_LEFT_X],
                y=bbox[BoundingBox.TOP_LEFT_Y],
                s=label_text,
                color='red'
            )

    plt.show()