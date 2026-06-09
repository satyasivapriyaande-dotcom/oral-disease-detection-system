import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 8


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    "dataset_cancer",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val = datagen.flow_from_directory(
    "dataset_cancer",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

base = DenseNet169(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation="relu")(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
class_weights = {
    0: 1.0,  
    1: 1.5    
}

model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    class_weight=class_weights
)

model.save("cancer_model.h5")
print("Cancer model saved")
