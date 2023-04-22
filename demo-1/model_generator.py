import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def generate_model(model_path, train_dataset, validation_dataset, input_shape, steps_per_epoch, epochs):
    # Define the model architecture
    model = keras.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_dataset)

    # Save the model
    model.save(model_path)

    return model