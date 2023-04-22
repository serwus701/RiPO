def extract_dataset(rescale, validation_split, src_path, input_shape, batch_size, class_mode):
    gen = ImageDataGenerator(rescale=rescale, validation_split=validation_split)

    # Create training and validation datasets
    train_dataset = gen.flow_from_directory(src_path,
                                            target_size=input_shape,
                                            batch_size=batch_size,
                                            class_mode=class_mode,
                                            subset='training')

    validation_dataset = gen.flow_from_directory(src_path,
                                                target_size=input_shape,
                                                batch_size=batch_size,
                                                class_mode=class_mode,
                                                subset='validation')
    
    return train_dataset, validation_dataset