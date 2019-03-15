from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator

model = MobileNetV2(input_shape=(64, 64, 3), alpha=0.25, weights=None, pooling='max', classes=3)

model.summary()

batch_size=32

image_generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=False,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5],
                               rescale=1./255,
                               validation_split=0.2)

train_generator = image_generator.flow_from_directory('dataset',
                                target_size=(64, 64, 3),
                                batch_size=batch_size,
                                subset='training')

validation_generator = image_generator.flow_from_directory('dataset',
                                target_size=(64, 64, 3),
                                batch_size=batch_size,
                                subset='val')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5)
    checkpointer = ModelCheckpoint(filepath='model_archive/sign-weights.{epoch:02d}-{val_loss:.5f}.h5', verbose=1, save_best_only=True)
    losses = model.fit_generator(train_generator,
                                 steps_per_epoch=train_generator.n // batch_size,
                                 validation_data=validation_generator,
                                 validation_steps=validation_generator.n // batch_size,
                                 verbose=1,
                                 epochs=20,
                                 callbacks=[early_stopping, checkpointer])