# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# gen = ImageDataGenerator(rescale=1/255,
#                             validation_split=0.2)

# input_shape = (500, 500)
# input_shape_2 = (500, 500, 3)

# path = 'human-detection-dataset'

# train_dataset = gen.flow_from_directory(path,
#                                             target_size=input_shape,
#                                             batch_size=32,
#                                             class_mode='binary',
#                                             subset='training')
                                            

                                        
# validation_dataset = gen.flow_from_directory(path,
#                                             target_size=input_shape,
#                                             batch_size=32,
#                                             class_mode='binary',
#                                             subset='validation')


# model = keras.Sequential([
#     layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape_2),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(train_dataset, steps_per_epoch = 3,  epochs=20, validation_data=validation_dataset)

# model.save('my_model')

# dir_path = 'human-detection-dataset'


# test_loss, test_acc= model.evaluate(validation_dataset,verbose=2)

# print('\nTest accuracy:', test_acc)
# print('\nTest loss:', test_loss)

import dataset_extractor
import model_generator

# Define constants
RESCALE = 1/255
VALIDATION_SPLIT = 0.2
INPUT_SHAPE = (500, 500)
INPUT_SHAPE_2 = (500, 500, 3)
BATCH_SIZE = 32
CLASS_MODE = 'binary'
EPOCHS = 20
STEPS_PER_EPOCH = 3
DATASET_PATH = 'human-detection-dataset'
MODEL_PATH = 'my_model'

train_dataset, validation_dataset = dataset_extractor.extract_dataset(RESCALE, VALIDATION_SPLIT, DATASET_PATH, INPUT_SHAPE, BATCH_SIZE, CLASS_MODE)
model = model_generator.generate_model(MODEL_PATH, train_dataset, validation_dataset, INPUT_SHAPE_2, STEPS_PER_EPOCH, EPOCHS)

# Evaluate the model on the validation dataset
test_loss, test_acc = model.evaluate(validation_dataset, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)