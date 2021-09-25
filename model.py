import glob
import numpy as np
import tensorflow
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout
from keras.optimizers import adam_v2
from sklearn.metrics import confusion_matrix, classification_report

train_dir = "/Users/dipit/Intel Image Classification/Data/seg_train"
test_dir = "/Users/dipit/Intel Image Classification/Data/seg_test"
classes = glob.glob('/Users/dipit/Intel Image Classification/Data/seg_train/*')
num_classes = len(classes)

img_rows, img_cols = 224, 224
batch_size = 32


def preprocess_image(train_dir, test_dir, img_rows, img_cols, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.25
    )
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="training"
    )
    val_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        subset="validation"
    )
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    return train_generator, val_generator, test_generator


train_data, val_data, test_data = preprocess_image(train_dir, test_dir, img_rows, img_cols, batch_size)


def build_model():
    basemodel = MobileNet(
        weights="imagenet",
        input_shape=(img_rows, img_cols, 3),
        include_top=False
    )
    for layer in basemodel.layers:
        layer.trainable = False
    x = basemodel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs=basemodel.input, outputs=output)
    return model


model = build_model()
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=adam_v2.Adam(learning_rate=0.0001),
    metrics=['acc']
)
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)
print("Evaluating the Model")
scores = model.evaluate(val_data, verbose=1)
print("Loss: ", scores[0])
print("Accuracy: ", scores[1])


keras.models.save_model(model, 'IntelImageClassifier.h5')
model = keras.models.load_model('IntelImageClassifier.h5')


print("Prediction on Validation")
val_pred = model.predict(val_data, verbose=1)
val_label = np.argmax(val_pred, axis=1)
class_labels = val_data.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print("Confusion Matrix")
print(confusion_matrix(val_data.classes, val_label))
print("Classification Report")
print(classification_report(val_data.classes, val_label, target_names=classes))


print("Prediction on Test data")
test_pred = model.predict(test_data, verbose=1)
test_label = np.argmax(test_pred, axis=1)
class_labels = test_data.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print("Confusion Matrix")
print(confusion_matrix(test_data.classes, test_label))
print("Classification Report")
print(classification_report(test_data.classes, test_label, target_names=classes))































