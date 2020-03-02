import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import os
import sys


def get_dict_bboxes(
        data_folder_path='D:\\workspace\\common\\Data Set\\Category and attribute prediction\\Category and Attribute Prediction Benchmark',
        eval_folder_name = 'Eval',
        anno_folder_name = 'Anno',
        list_category_img="list_category_img.txt",
        list_eval_partition='list_eval_partition.txt',
        list_bbox="list_bbox.txt"
):

    with open(os.path.join(data_folder_path, anno_folder_name, list_category_img), 'r') as category_img_file,\
         open(os.path.join(data_folder_path, eval_folder_name, list_eval_partition), 'r') as eval_partition_file,\
         open(os.path.join(data_folder_path, anno_folder_name, list_bbox), 'r') as bbox_file:

        list_category_img = [line.rstrip('\n').split() for line in category_img_file][2:]
        list_category_img = {item[0]: item[1] for item in list_category_img}

        list_eval_partition = [line.rstrip('\n').split() for line in eval_partition_file][2:]
        list_eval_partition = {item[0]: item[1] for item in list_eval_partition}

        list_bbox = [line.rstrip('\n').split() for line in bbox_file][2:]
        list_bbox = {item[0]:item[1:] for item in list_bbox}

        list_all = {key: {
            "category": list_category_img.get(key),
            "dataset": list_eval_partition.get(key),
            "bounding_box": [int(coordinate) for coordinate in list_bbox.get(key)]
        } for key in list_category_img.keys()}

        train_set = [
            {
                "path": key,
                "category": value["category"],
                "bounding_box": value["bounding_box"]
            } for key, value in list_all.items() if value["dataset"] == "train"
        ]

        dev_set = [
            {
                "path": key,
                "category": value["category"],
                "bounding_box": value["bounding_box"]
            } for key, value in list_all.items() if value["dataset"] == "val"
        ]

        test_set = [
            {
                "path": key,
                "category": value["category"],
                "bounding_box": value["bounding_box"]
            } for key, value in list_all.items() if value["dataset"] == "test"
        ]

        return train_set, dev_set, test_set


def load_image(path):
    img = keras.preprocessing.image.load_img(path=path)
    img = np.array(img)
    return img


def show_image(image, bbox, label):
    # Show image
    plt.imshow(image)

    # Draw bounding box
    ax = plt.gca()
    rect = Rectangle(
        xy=(bbox["top_left_x"], bbox["top_left_y"]),
        width=bbox["bottom_right_x"] - bbox["top_left_x"],
        height=bbox["bottom_right_y"] - bbox["top_left_y"],
        fill=False,
        color='red')
    ax.add_patch(rect)

    # Write label
    label_text = "Label = {}".format(label)
    plt.text(
        x=bbox["top_left_x"],
        y=bbox["top_left_y"],
        s=label_text,
        color='red'
    )
    plt.show()


def normalize_image(image):
    image = image/255.
    return image


def data_generator(data_set, data_folder_path='D:\\workspace\\common\\Data Set\\Category and attribute prediction\\Category and Attribute Prediction Benchmark'):
    for index, example in enumerate(data_set):
        img = load_image(os.path.join(data_folder_path, example["path"]))
        label = int(example["category"])
        bbox = example["bounding_box"]
        bbox = {
            "top_left_x": bbox[0],
            "top_left_y": bbox[1],
            "bottom_right_x": bbox[2],
            "bottom_right_y": bbox[3]
        }
        yield img, label, bbox

    return





data_folder_path='D:\\workspace\\common\\Data Set\\Category and attribute prediction\\Category and Attribute Prediction Benchmark'
train_set, dev_set, test_set = get_dict_bboxes(data_folder_path)
train_generator = data_generator(data_set=train_set)

for index, (img, label, bbox) in enumerate(train_generator):

    img = image.apply_affine_transform(
        theta=0, # rotation
        tx=0, # Translate X
        ty=0, # Translate Y
        shear=0, # Shear
        zx=1, # Scale X
        zy=1, # Scale Y
        row_axis=0, # axis of row
        col_axis=1, # axis of col
        channel_axis=2, # axis of color channel
        fill_mode='constant', # missing pixel fill mode
        cval=0.,
        order=1
    )
    show_image(image=img, bbox=bbox, label=label)
    print(type(img))
    print(img.shape)
    print(label)
    print(bbox)
    break




