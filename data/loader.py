import os


def get_deep_fashion_annotations(data_folder_path='D:\workspace\common\Data Set\Category and attribute prediction\Category and Attribute Prediction Benchmark'):
    eval_folder_name = 'Eval'
    anno_folder_name = 'Anno'

    with open(os.path.join(data_folder_path, anno_folder_name, 'list_category_img.txt'), 'r') as category_img_file,\
         open(os.path.join(data_folder_path, eval_folder_name, 'list_eval_partition.txt'), 'r') as eval_partition_file,\
         open(os.path.join(data_folder_path, anno_folder_name, 'list_bbox.txt'), 'r') as bbox_file:

        list_category_img = [line.rstrip('\n').split() for line in category_img_file][2:]
        list_category_img = {item[0]: item[1] for item in list_category_img}

        list_eval_partition = [line.rstrip('\n').split() for line in eval_partition_file][2:]
        list_eval_partition = {item[0]: item[1] for item in list_eval_partition}

        list_bbox = [line.rstrip('\n').split() for line in bbox_file][2:]
        list_bbox = {item[0]:item[1:] for item in list_bbox}

        list_all = {os.path.join(data_folder_path, key): {
            "category": int(list_category_img.get(key)),
            "dataset": list_eval_partition.get(key),
            "bounding_box": [int(list_bbox.get(key)[i]) for i in range(len(list_bbox.get(key)))]
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

def get_deep_fashion_annotations_by_category_type(data_folder_path='D:\workspace\common\Data Set\Category and attribute prediction\Category and Attribute Prediction Benchmark'):
    eval_folder_name = 'Eval'
    anno_folder_name = 'Anno'

    with open(os.path.join(data_folder_path, anno_folder_name, 'list_category_cloth.txt'), 'r') as category_img_file,\
         open(os.path.join(data_folder_path, eval_folder_name, 'list_eval_partition.txt'), 'r') as eval_partition_file,\
         open(os.path.join(data_folder_path, anno_folder_name, 'list_bbox.txt'), 'r') as bbox_file:

        list_category_img = [line.rstrip('\n').split() for line in category_img_file][2:]
        list_category_img = {item[0]: item[1] for item in list_category_img}

        list_eval_partition = [line.rstrip('\n').split() for line in eval_partition_file][2:]
        list_eval_partition = {item[0]: item[1] for item in list_eval_partition}

        list_bbox = [line.rstrip('\n').split() for line in bbox_file][2:]
        list_bbox = {item[0]:item[1:] for item in list_bbox}

        list_all = {os.path.join(data_folder_path, key): {
            "category": int(list_category_img.get(key.split("/")[1].split("_")[-1])),
            "dataset": list_eval_partition.get(key),
            "bounding_box": [int(list_bbox.get(key)[i]) for i in range(len(list_bbox.get(key)))]
        } for key in list_eval_partition.keys()}

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