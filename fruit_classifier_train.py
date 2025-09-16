#importing libraries
from tensorflow.keras.applications import MobileNet
import pandas as pd,numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers,optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau


#using Imagedatgenerator for data augmentation in train,test, and validation  i.e creating more training data by resizing,
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1/255)

#paths for directory
train_dir = r"archive (1)\dataset-4\train"
val_dir   = r"archive (1)\dataset-4\test"

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224,224),
    batch_size=32, class_mode='sparse'
)

print(train_gen.class_indices) # Print class indices for training data

valid_gen = valid_datagen.flow_from_directory(
    val_dir,   target_size=(224,224),
    batch_size=32, class_mode='sparse'
)

#created list if wanted to make predictions
fruit_arr = [k for k, v in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]
print("Aligned fruit_arr:", fruit_arr)

#neural network bulding, merging pretrained model with CNN
def build_model(optimizer='adam', learning_rate=1e-5, dropout_rate=0.1, dense_units=32, unfreeze_layers=27):
    base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    
    base_model.trainable = True
    total_layers = len(base_model.layers)
    for layer in base_model.layers[:total_layers - unfreeze_layers]:
        layer.trainable = False

 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')  
    ])

    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



train_gen.reset()

final_model = build_model()

#compiling the model
final_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='rmsprop', 
    metrics=['accuracy']
)

#prevention from overfitting
earlystop_cb = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)


train_gen.reset()
valid_gen.reset()

#fitting the model
history = final_model.fit(
    train_gen,
    steps_per_epoch = len(train_gen), 
    epochs=27, 
    batch_size=32,  
    validation_data=valid_gen,
    validation_steps=None,
    callbacks=[checkpoint_cb, earlystop_cb,reduce_lr]
)

#visualization
df = pd.DataFrame(history.history)
df[['loss','val_loss']].plot()
plt.title('Training and Validation Loss')
plt.show()

df[['accuracy','val_accuracy']].plot()
plt.title('Training and Validation Accuracy')
plt.show()

#saving the model
final_model.save(r'archive (1)\dataset-4\fruit_tf_model.h5')
