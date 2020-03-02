import tensorflow.keras as keras
import numpy as np
from PIL import Image
from config.enum import ObjDetection
import data.normalize as normalize


class DFImageGeneratorByPath(keras.utils.Sequence):
    def __init__(
            self,
            dataset: list,
            batch_size: int = 16,
            target_size=(256, 256),
            n_channel=3,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_channel = n_channel
        self.augmenter = DERandAugment()
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

            image, annotations = normalize.resize(image=image, annotations=annotations, target_size=self.target_size)
            image, annotations = self.augmenter.transform(image=image, annotations=annotations)
            image, annotations = normalize.scale_to_unit(image=image, annotations=annotations)

            batch_images[i] = image
            batch_label[i] = annotations[0][ObjDetection.LABEL]
            batch_bbox[i] = annotations[0][ObjDetection.BBOX]

        return batch_images, [batch_label, batch_bbox]

    def on_epoch_end(self):
        return


import utils.visualize as visualize
import data.loader as loader
from augmentation.transform import DERandAugment
from tensorflow.keras.applications.resnet50 import ResNet50



train_set, dev_set, test_set = loader.get_deep_fashion_annotations()

train_gen = DFImageGeneratorByPath(
    train_set
)

dev_gen = DFImageGeneratorByPath(
    dev_set
)

test_gen = DFImageGeneratorByPath(
    test_set
)


def build_model():
    model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    for layer in model_resnet.layers[:-12]:
        # 6 - 12 - 18 have been tried. 12 is the best.
        layer.trainable = False

    x = model_resnet.output # (None, 2048)
    # print(len(model_resnet.layers))
    # return
    x = keras.layers.Dense(512, activation='elu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    y = keras.layers.Dense(60, activation='softmax', name='img')(x)

    x_bbox = model_resnet.output
    x_bbox = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x_bbox)
    x_bbox = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x_bbox)
    bbox = keras.layers.Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

    final_model = keras.models.Model(
        inputs=model_resnet.input,
        outputs=[y, bbox]
    )

    opt = keras.optimizers.Adam(lr=0.0001)
    final_model.compile(optimizer=opt,
                        loss={
                            'img': 'sparse_categorical_crossentropy',
                            'bbox': 'mean_squared_error'
                        },
                        metrics={
                            'img': ['accuracy', 'top_k_categorical_accuracy'],  # default: top-5
                            'bbox': ['mse']
                        }
                        )
    return final_model


build_model()

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
# # checkpoint = keras.callbacks.ModelCheckpoint('./models/model.h5')
#
final_model = build_model()
final_model.fit_generator(
    train_gen,
    epochs=24,
    validation_data=dev_gen,
    verbose=1,
    shuffle=True,
    callbacks=[lr_reducer, early_stopper],
    workers=1
)

final_model.save("demo_detector.h5")

print("FINISH TRAINING")

scores = final_model.evaluate_generator(
    test_gen,
    verbose=1
)

# augmenter = DERandAugment()

# for images, labels, bboxs in generator:
#     print(bboxs.shape)
#     for i in range(len(images)):
#         image = images[i]
#         annotations = [
#                 {
#                     ObjDetection.BBOX: bboxs[i],
#                     ObjDetection.LABEL: labels[i],
#                 }
#             ]






