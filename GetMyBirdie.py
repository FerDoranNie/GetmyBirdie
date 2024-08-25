import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout , GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint



train_dir = './Birds_Species_Dataset/train'
validation_dir = './Birds_Species_Dataset/valid'
BATCH_SIZE = 64
IMG_SIZE = (224, 224)


image_path = "Birds_Species_Dataset/train/AMERICAN AVOCET/001.jpg"

image = Image.open(image_path)

width, height = image.size
channels = len(image.getbands())  # Esto debería devolver 3 para RGB

print(f"Width: {width}, Height: {height}, Channels: {channels}")

# Verificar si la imagen tiene las dimensiones correctas
if (width, height) == (224, 224) and channels == 3:
    print("La imagen tiene el tamaño correcto (224, 224, 3)")
else:
    print("La imagen no tiene el tamaño correcto")
    

for filename in os.listdir(train_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(train_dir, filename)
        image = Image.open(image_path)
        width, height = image.size
        channels = len(image.getbands())
        
        if (width, height) != (224, 224) or channels != 3:
            print(f"Imagen {filename} no tiene el tamaño correcto: {width}x{height}x{channels}")


train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


image_path = "Birds_Species_Dataset/train/AMERICAN AVOCET/001.jpg"
image = load_img(image_path)
image_array = img_to_array(image)
image_array = tf.expand_dims(image_array, axis=0)  # Expande la dimensión para simular un batch

# Aplicar augmentación de datos
augmented_image = data_augmentation(image_array)

# Convertir de nuevo a imagen para visualizar
augmented_image = array_to_img(augmented_image[0])
augmented_image.show()  # Esto debería mostrar la imagen con augmentación aplicada

random_flip = tf.keras.layers.RandomFlip('horizontal')
random_rotation = tf.keras.layers.RandomRotation(0.2)


for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[39]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')


rescale = tf.keras.layers.Rescaling(1./255)


mobilenet = MobileNetV2( include_top=False, 
                         weights="imagenet", 
                         input_shape=(224,224,3))

mobilenet.build(input_shape=(None, 224, 224, 3))


mobilenet.trainable=False


for layer in mobilenet.layers[:-20]:
    layer.trainable = True


testmobilenet = MobileNetV2(include_top=False, 
                            weights="imagenet", 
                            input_shape=(224, 224, 3))

simple_model = Sequential([
    testmobilenet,
    GlobalAveragePooling2D(),
    Dense(450, activation='softmax')  # Ajusta '450' al número de clases en tu dataset
])

simple_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simple_model.summary()


simple_model2 = Sequential([
    rescale,
    testmobilenet,
    GlobalAveragePooling2D(),
    #Dense(450, activation='softmax')  # Ajusta '450' al número de clases en tu dataset,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(450, activation='softmax')
])

simple_model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simple_model2.summary()


def preprocess_dataset(dataset):
    return dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)


train_dataset = preprocess_dataset(train_dataset)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Métrica a monitorear (usualmente val_loss o val_accuracy)
    patience=10,  # Número de épocas sin mejora antes de detener el entrenamiento
    restore_best_weights=True  # Restaura los pesos del modelo a la mejor época
)


checkpoint_filepath = './Birds_Species_Dataset/bird_model.keras'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,  # Ruta donde guardar el modelo
    monitor='val_loss',  # Métrica a monitorear
    save_best_only=True,  # Solo guarda el modelo si es el mejor encontrado
    save_weights_only=False,  # Si solo quieres guardar los pesos, puedes poner esto en True
    mode='min'  # Modo min para val_loss, max para val_accuracy
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Métrica a monitorear
    factor=0.2,  # Factor por el cual se reduce la tasa de aprendizaje
    patience=5,  # Número de épocas sin mejora antes de reducir la tasa de aprendizaje
    min_lr=1e-6  # Tasa de aprendizaje mínima
)


tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',  # Directorio donde guardar los archivos de logs
    histogram_freq=1,  # Frecuencia (en épocas) para calcular histogramas
    write_graph=True,  # Guardar el gráfico del modelo
    write_images=True  # Guardar imágenes de los pesos
)


callbacks = [early_stopping, model_checkpoint, reduce_lr, tensorboard]

train_labels = [label for _, label in train_dataset]

non_integer_labels = [label for label in train_labels if not isinstance(label, int)]
if non_integer_labels:
    print(f"Etiquetas no enteras encontradas: {non_integer_labels[:10]}")
else:
    print("Todas las etiquetas son enteras.")
    
train_labels = [label for _, label in train_dataset]


MobileNetV2Model = Sequential([
    #data_augmentation,
    rescale,
    mobilenet,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Flatten(),
    #Dense(512, activation='relu'),
    #Dense(num_classes, activation='relu'),
    Dense(540, activation='relu'),    
    BatchNormalization(),
    Dropout(0.3),
    Dense(540, activation='softmax')
])

MobileNetV2Model.compile( optimizer="adamax", 
                          loss="sparse_categorical_crossentropy" , 
                          metrics=["accuracy"] )


batch_size= 32
steps_per_epoch = len(train_dataset) // batch_size
validation_steps = len(validation_dataset) // batch_size
steps_per_epoch2 = int(len(train_dataset) * 0.5)
validation_steps2 = int(len(validation_dataset) *0.5)

steps_per_epoch3 = int(len(train_dataset) * 1)
validation_steps3 = int(len(validation_dataset) *1)

print(steps_per_epoch)
print(validation_steps)
print(steps_per_epoch2)
print(validation_steps2)
print(steps_per_epoch3)
print(validation_steps3)


history = MobileNetV2Model.fit(
                               #train_dataset,
                               train_dataset.repeat(),
                               epochs=15 , 
                               #epochs=4,
                               batch_size=32 ,
                               #steps_per_epoch = len(train_dataset),
                               steps_per_epoch = steps_per_epoch,
                               callbacks=callbacks,
                               #workers=10, 
                               validation_data=validation_dataset,
                               #validation_steps = len(validation_dataset)
                               validation_steps = validation_steps
                               )
