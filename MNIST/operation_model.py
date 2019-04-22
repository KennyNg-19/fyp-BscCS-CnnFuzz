# -*- coding: utf-8 -*-

import keras

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_datas, train_labels, name_of_file=None, epochs=20, \
                batch_size=40, verbose=False, with_checkpoint=False):
    if with_checkpoint:
        prefix = ''
        filepath = prefix + name_of_file + '-{epoch:02d}-{loss:.4f}.h5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=5, \
            save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, \
            callbacks=callbacks_list, verbose=0)
    else:
        # batch_size, default to 32
        model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=None, verbose=0) #
    return model