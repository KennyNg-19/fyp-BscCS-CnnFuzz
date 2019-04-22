# functions for I/O and initialization(preprocess)
import random
import os
import keras
import cv2
import numpy as np
from collections import defaultdict
from keras.preprocessing import image
from keras import backend as K

img_height, img_width = 28, 28

def shuffle_in_uni(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    length = len(a)
    permutation = np.random.permutation(length)
    index_permutation = np.arange(length)
    shuffled_a[permutation] = a[index_permutation]
    shuffled_b[permutation] = b[index_permutation]
    return shuffled_a, shuffled_b

def load_and_preprocess_data(number_of_train_data, number_of_test_data):
    (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()
    
    (train_datas, train_labels) = shuffle_in_uni(train_datas, train_labels)
    (test_datas, test_labels) = shuffle_in_uni(test_datas, test_labels)
    print("Shuffle train_datas and test_datas,")

    train_labels = train_labels[:number_of_train_data]
    test_labels = test_labels[:number_of_test_data]
    train_datas = train_datas[:number_of_train_data]
    test_datas = test_datas[:number_of_test_data]
    print("and extract: training " + str(number_of_train_data) + " + testing " + str(number_of_test_data))

    train_datas = preprocess_data(img_width, img_height, train_datas)
    test_datas = preprocess_data(img_width, img_height, test_datas)

    # One-hot encoding the labels
    train_labels = keras.utils.np_utils.to_categorical(train_labels)
    test_labels = keras.utils.np_utils.to_categorical(test_labels)
    return (train_datas, train_labels), (test_datas, test_labels)

def load_file(file):
    if os.path.exists(file):
        print("%s loaded" % file)
        return np.load(file).item()
    else:
        print('No %s exists' % file)
        os._exit(0)

def preprocess_data(width_without_padding, height_without_pading, datas):
    datas = datas.reshape(datas.shape[0], width_without_padding, height_without_pading, 1)

    mean_px = datas.mean().astype(np.float32)
    std_px = datas.std().astype(np.float32)
    datas = (datas - mean_px) / (std_px)

    return datas


def init_storage_dir(save_dir):
    if os.path.exists(save_dir):
        print("dir exists ! files inside will be removed !")
        # remove files in dir
        for i in os.listdir(save_dir):
            path_file = os.path.join(save_dir, i)
            if os.path.isfile(path_file):
                os.remove(path_file)

    # if storage dir not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# Start Fuzzing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    return input_img_data

def deprocess_image(x):
    # de-normalization: [0,1] -> [0,255]
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2]) 


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)




# --------------------------------------------------
def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


# dict.value：single value, 0 / 1+
def init_neuron_coverage(model, model_layer_times): 
    for layer in model.layers:
        # 对于不经过activation的layer, 不考虑其coverage
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        # 对于经过activation的layer
        for index in range(layer.output_shape[-1]):
        # lth layer nth Neuron: tuple(layer.name, index)is key, referring to a neuron
            model_layer_times[(layer.name, index)] = 0 

def init_coverage_times(model):
    model_layer_times = defaultdict(int)
    init_neuron_coverage(model,model_layer_times)
    return model_layer_times

def init_coverage_value(model):
    model_layer_value = defaultdict(float)
    init_neuron_coverage(model, model_layer_value)
    return model_layer_value

def init_neuron_values(model):
    model_neuron_values = defaultdict(float)
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]): 
            # lth layer nth Neuron: tuple(layer.name, index)is key, referring to a neuron
            model_neuron_values[(layer.name, index)] = [0, 0]
    return model_neuron_values

# dict.value： multiple values in a list, 0 / 1+
def init_multisection_coverage_value(model, multisection_num):
    neurons_multisection_coverage_values = defaultdict(float)
    for layer in model.layers:
        
        if 'flatten' in layer.name or 'input' in layer.name:
            continue        
        for index in range(layer.output_shape[-1]): # 输出张量 last D

            neurons_multisection_coverage_values[(layer.name, index)] = [0] * multisection_num # [0,0,0,....]
    return neurons_multisection_coverage_values