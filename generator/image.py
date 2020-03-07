import tensorflow.keras as keras
import numpy as np
from PIL import Image
from config.enum import ObjDetection
import data.normalize as normalize


class DFImageGeneratorByPath(keras.utils.Sequence):
    def __init__(
            self,
            dataset: list,
            batch_size: int = 8,
            target_size=(256, 256),
            n_channel=3,
            classes: list = [],
            need_augment=False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_channel = n_channel
        self.augmenter = DERandAugment()
        self.classes = classes
        self.need_augment = need_augment
        return

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]

        n_data = len(batch)

        batch_images = np.zeros(shape=(n_data,) + self.target_size + (self.n_channel,))
        batch_label = np.zeros(shape=n_data)
        batch_bbox = np.zeros((n_data,) + (4,), dtype=np.float)

        for i in range(n_data):
            image = Image.open(batch[i]["path"])
            annotations = [
                {
                    ObjDetection.BBOX: batch[i]["bounding_box"],
                    ObjDetection.LABEL: batch[i]["category"]
                }
            ]

            # image = image.crop(annotations[0][ObjDetection.BBOX])
            image, annotations = normalize.resize(image=image, annotations=annotations, target_size=self.target_size)
            if self.need_augment:
                image, annotations = self.augmenter.transform(image=image, annotations=annotations)

            image, annotations = normalize.scale_to_unit(image=image, annotations=annotations)


            batch_images[i] = image
            batch_label[i] = self.classes.index(annotations[0][ObjDetection.LABEL])
            batch_bbox[i] = annotations[0][ObjDetection.BBOX]

        return batch_images, batch_label
        # return batch_images, [batch_label, batch_bbox]

    def on_epoch_end(self):
        return


import utils.visualize as visualize
import data.loader as loader
from augmentation.transform import DERandAugment
from tensorflow.keras.applications.densenet import DenseNet121


train_set, dev_set, test_set = loader.get_deep_fashion_annotations_by_category_type()
# train_set, dev_set, test_set = loader.get_deep_fashion_annotations()

classes = [item["category"] for item in train_set] + [item["category"] for item in dev_set] + [item["category"] for item in test_set]
classes = np.array(list(set(classes)))
np.save("classes.npy", classes)
classes = classes.tolist()

train_gen = DFImageGeneratorByPath(
    train_set,
    classes=classes,
    need_augment=True
)

dev_gen = DFImageGeneratorByPath(
    dev_set,
    classes=classes
)

test_gen = DFImageGeneratorByPath(
    test_set,
    classes=classes
)


def build_model():
    model_resnet = DenseNet121(include_top=False, pooling='avg')
    # model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # for layer in model_resnet.layers[:-12]:
        # 6 - 12 - 18 have been tried. 12 is the best.
        # print(layer)
        # layer.trainable = False

    x = model_resnet.output # (None, 2048)
    # print(len(model_resnet.layers))
    # return
    # x = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    y = keras.layers.Dense(3, activation='softmax', name='img')(x)

    # x_bbox = model_resnet.output
    # x_bbox = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x_bbox)
    # x_bbox = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x_bbox)
    # bbox = keras.layers.Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

    final_model = keras.models.Model(
        inputs=model_resnet.input,
        outputs=y
        #outputs=[y, bbox]
    )

    opt = keras.optimizers.Adam()
    # opt = keras.optimizers.Adam(lr=0.0001, )
    final_model.compile(optimizer=opt,
                        loss={
                            'img': 'sparse_categorical_crossentropy',
                    #        'bbox': 'mean_squared_error'
                        },
                        metrics={
                            'img': [
                                'accuracy',
                                # 'sparse_top_k_categorical_accuracy'
                            ],  # default: top-5
                     #       'bbox': ['mse']
                        }
                        )
    return final_model


def experiment():

    lr_reducer = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=12,
        factor=0.5,
        verbose=1
    )

    # # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=30,
                                                  verbose=1)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5')
    final_model = build_model()
    # final_model.evaluate(dev_gen)
    # return
    #final_model.load_weights("model.h5")
    final_model.fit_generator(
        train_gen,
        epochs=200,
        validation_data=dev_gen,
        verbose=1,
        shuffle=True,
        callbacks=[lr_reducer, early_stopper, checkpoint],
        workers=1
    )

    final_model.save("demo_detector.h5")

    print("FINISH TRAINING")

    scores = final_model.evaluate_generator(
        test_gen,
        verbose=1
    )

    return


def test_model():
    model = keras.models.load_model("model.h5")
    model.evaluate_generator(dev_gen, verbose=1)

    for example in train_gen:
        result = model.predict(example[0][0:1])
        print(result)
        visualize.show_image(
            image=example[0][0],
            annotations=[
                {
                    ObjDetection.BBOX: result[1][0]*256
                }
            ]
        )
        break
    return


experiment()
test_model()

# augmenter = DERandAugment()
print(len(train_set), len(dev_set), len(test_set))

for images, labels in DFImageGeneratorByPath(
# for images, (labels, bboxs) in DFImageGeneratorByPath(
    dataset=dev_set,
    classes=classes
):
    for i in range(len(images)):
        image = images[i]
        annotations = [
                {
                    ObjDetection.BBOX: [0,0,0,0],
                    ObjDetection.LABEL: labels[i],
                }
            ]

        visualize.show_image(image, annotations)
        # image = image.crop(annotations[0][ObjDetection.BBOX])
        # visualize.show_image(image, annotations)






