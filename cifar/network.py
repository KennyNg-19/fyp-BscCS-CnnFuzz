import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras
import random


class CNNNetwork():

    def __init__(self):
        self.number_of_train_data = int(5000 * 0.6)
        self.number_of_test_data = int(1000 * 0.6)
        self.width_without_padding = 28
        self.height_without_pading = 28
        self.width_with_padding = 32
        self.height_with_padding = 32
        self.num_of_channels = 1

    def shuffle_in_uni(self, a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        length = len(a)
        permutation = np.random.permutation(length)
        index_permutation = np.arange(length)
        shuffled_a[permutation] = a[index_permutation]
        shuffled_b[permutation] = b[index_permutation]
        return shuffled_a, shuffled_b

    def load_and_preprocess_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()
        # Truncate 5000 training samples and 1000 test samples apart from original dataset
        (train_datas, train_labels) = self.shuffle_in_uni(train_datas, train_labels)
        (test_datas, test_labels) = self.shuffle_in_uni(test_datas, test_labels)
        print("Shuffle train_datas and test_datas,")

        train_labels = train_labels[:self.number_of_train_data]
        test_labels = test_labels[:self.number_of_test_data]
        train_datas = train_datas[:self.number_of_train_data]
        test_datas = test_datas[:self.number_of_test_data]
        print("and extract: training " + str(self.number_of_train_data) + " + testing " + str(self.number_of_test_data))

        train_datas = self.preprocess_data(train_datas)
        test_datas = self.preprocess_data(test_datas)

        # One-hot encoding the labels
        train_labels = keras.utils.np_utils.to_categorical(train_labels)
        test_labels = keras.utils.np_utils.to_categorical(test_labels)
        return (train_datas, train_labels), (test_datas, test_labels)

    def preprocess_data(self, datas):
        datas = datas.reshape(datas.shape[0], self.width_without_padding, self.height_without_pading, 1)

        # Pad on the original images, from 28 * 28 to 32 * 32
        # datas = np.pad(datas, ((0,0),(2,2),(2,2),(0,0)), 'constant')

        # Standardize training samples
        mean_px = datas.mean().astype(np.float32)
        std_px = datas.std().astype(np.float32)
        datas = (datas - mean_px) / (std_px)

        return datas

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)

    # LeNet-5
    def create_CNN_model_1(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=6,
                                kernel_size=5,
                                strides=1,
                                activation='relu',
                                input_shape=(self.width_with_padding, self.height_with_padding, self.num_of_channels)),
            keras.layers.MaxPooling2D(pool_size=2, strides = 2),
            keras.layers.Conv2D(filters=16,
                                kernel_size=5,
                                strides=1,
                                activation='relu',
                                input_shape=(14, 14, 6)),
            keras.layers.MaxPooling2D(pool_size=2, strides = 2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=120, activation='relu'),
            keras.layers.Dense(units=84, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        return model

    # Adversarial Networks: Generating Adversarial Examples
    def create_CNN_model_2(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(self.width_with_padding, self.height_with_padding, self.num_of_channels)),
            keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(30, 30, 32)),
            keras.layers.MaxPooling2D(pool_size=2, strides = 2),
            keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(14, 14, 32)),
            keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(12, 12, 32)),
            keras.layers.MaxPooling2D(pool_size=2, strides = 2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=200, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        return model

    def compile_model(self, model):
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_datas, train_labels, name_of_file=None, epochs=30, batch_size=32, verbose=False, data_augmentation=True):
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(train_datas, train_labels,
                      batch_size=batch_size,
                      epochs=epochs,
                    #   validation_data=(x_test, y_test),
                      verbose = 0,
                      shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(train_datas)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(train_datas, train_labels,
                                             batch_size=batch_size),
                                epochs=epochs,
                                verbose= 0,
                                # validation_data=(x_test, y_test),
                                workers=4)


            return model

    def evaluate_model(self, model, test_datas, test_labels, mode='normal'):
        loss, acc = model.evaluate(test_datas, test_labels)
        if mode == 'normal':
            print('Normal model accurancy: {:5.2f}%'.format(100*acc))
            print('')
        else:
            print(mode, 'mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

    def save_model(self, model, name_of_file, mode='normal'):
        prefix = ''
        file_name = prefix + name_of_file + '.h5'
        model.save(file_name)
        if mode == 'normal':
            print('Normal model is successfully trained and saved at', file_name)
        else:
            print('Mutated model by ' + mode + ' is successfully saved at', file_name)
        print('')

    def train_and_save_simply_CNN_model(self, name_of_file=None, verbose=False, with_checkpoint=False, model_index=1):
        (train_datas, train_labels), (test_datas, test_labels) = self.load_data()
        if model_index == 1:
            model = self.create_CNN_model_1()
        else:
            model = self.create_CNN_model_2()

        model = self.compile_model(model)
        model = self.train_model(model, train_datas, train_labels, verbose=verbose, with_checkpoint=with_checkpoint)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_datas.shape)
            print('test dataset shape:', test_datas.shape)
            print('network architecture:')
            model.summary()
            print('')

            self.evaluate_model(model, test_datas, test_labels)

        self.save_model(model, 'CNN_model'+str(model_index))
