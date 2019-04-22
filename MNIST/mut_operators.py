import tensorflow as tf
import numpy as np
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator
from operation_model import compile_model, train_model

import random
import math
import utils
from IO_and_init import *

total_test_cases_num = 0

def run_operator(mutation_ratios, operator_name, model, train_dataset, test_datas, test_labels, \
    save_dir, AFRs_mutated_layer_indices = None):
    
    global total_test_cases_num # will be modified
    # local var
    (train_datas, train_labels) = train_dataset
    gen_test_cases = np.zeros(train_dataset[0].shape) # train_datas 是 global var
    gen_test_cases = np.delete(gen_test_cases, slice(0, gen_test_cases.shape[0]), axis=0) # remove the all the lines, only keeps the shape (0, 32, 32, 1)
    difference_indexes = [] # a set for indexes with differences
    counter = 0

    right_labels = np.argmax(test_labels, axis=1) # right class labels

    print("\n-----------------------------" + operator_name + " mutation operator-----------------------------")
    for mutation_ratio in mutation_ratios:
        if operator_name == 'DR':
            (mutated_datas, mutated_labels), mutated_model = DR_mut(train_dataset, model, mutation_ratio)
        elif operator_name == 'LE':
            lower_bound = 0
            upper_bound = 9
            (mutated_datas, mutated_labels), mutated_model = LE_mut(train_dataset, model, lower_bound, upper_bound, mutation_ratio)
        elif operator_name == 'DM':
            (mutated_datas, mutated_labels), mutated_model = DM_mut(train_dataset, model, mutation_ratio)
        elif operator_name == 'DF':
            (mutated_datas, mutated_labels), mutated_model = DF_mut(train_dataset, model, mutation_ratio)
        elif operator_name == 'NP':
            STD = 5
            (mutated_datas, mutated_labels), mutated_model = NP_mut(train_dataset, model, mutation_ratio, STD=STD)
        elif operator_name == 'AFRs':
            mutated_layer_indices = AFRs_mutated_layer_indices
            (mutated_datas, mutated_labels), mutated_model = AFRs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif operator_name == 'whi' or operator_name == 'rot' or operator_name == 'sh' or operator_name == 'fl':
            (mutated_datas, mutated_labels), mutated_model = aug_mut(train_dataset, model, mutation_ratio, operator_name)
        else:
            print("Input is not a valid operator mode")
            return

        # compile model
        trained_model = compile_model(model)
        mutated_model = compile_model(mutated_model)

        # train model
        trained_model = train_model(model, train_datas, train_labels)
        trained_mutated_model = train_model(mutated_model, mutated_datas, mutated_labels)

        # evaluate model and get accurracy
        # loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        # normal_accs.append(acc)
        # mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
        # mutant_accs.append(mutant_acc)

        # get the min and max of each neuron in original model! store in a dict
    # if use_other_metric:

        # ---------------------------------find different behaviors---------------------------------
        # quality control of mutant model
        trained_loss, trained_acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
        if mutant_acc < trained_acc - 0.03 * (counter + 1):
          print("\n{0}th bad mutant with low acc {1:.2%} < {2:.2%}, mutation ratio {3:.3f}, will be dropped out\n".format(counter + 1, \
          mutant_acc, trained_acc - 0.03 * (counter + 1), mutation_ratio))
          continue
        print("\n{0}th mutant passes the quality test, with acc {1:.2%} >= {2:.2%} and mutation ratio {3:.3f}".format(counter + 1, \
        mutant_acc, trained_acc - 0.03 * (counter + 1), mutation_ratio))

        # test the mutated model
        origit_predi_labels = np.argmax(trained_model.predict(test_datas), axis = 1)
        mutant_predi_labels = np.argmax(trained_mutated_model.predict(test_datas), axis = 1)

        # compare the test reasults with correct result
        #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
        difference_indexes = difference_indexes + list(np.nonzero(origit_predi_labels - mutant_predi_labels)[0])
        difference_score = len(difference_indexes) / origit_predi_labels.size

        # collect all selected test cases
        prev_gen_test_cases = gen_test_cases
        additional_test_cases = test_datas[np.nonzero(origit_predi_labels - mutant_predi_labels)[0]]
        concat_test_cases = np.append(gen_test_cases, additional_test_cases,axis = 0)
        _, idx = np.unique(concat_test_cases, axis = 0, return_index=True)
        gen_test_cases = concat_test_cases[np.sort(idx)]

        counter += 1
        print("New test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))
        # end of loop

    # save the test cases causing differences
    store_test_cases(save_dir, operator_name, difference_indexes, right_labels, gen_test_cases)

    total_test_cases_num += gen_test_cases.shape[0]
    print("-----------------------now %d test cases are saved-----------------------\n" % total_test_cases_num)


# ----------------------------------------------------------------------------
def remove_duplicate_without_sorting(seq):
   # order preserving
   noDupes = []
   [noDupes.append(i) for i in seq if not noDupes.count(i)]
   return noDupes

# store test cases in to local folder
# difficult: correspondences exist between test cases and their respecitve real labels
def store_test_cases(save_dir, mutation_name, difference_indexes, right_labels, gen_test_cases):
    difference_indexes = remove_duplicate_without_sorting(difference_indexes)
    for index, test_case in enumerate(gen_test_cases):
        cv2.imwrite('%s%d_%d.png' % (save_dir + mutation_name, index + 1, \
        right_labels[difference_indexes[index]]), deprocess_image(test_case.reshape(1,28,28,1)))


# 0. different data augmentation configurations
# includes, featurewise_center, zca_whitening, rotation_range, horizontal_flip=True, vertical_flip=True)
def aug_mut(train_dataset, model, mutation_ratio, mut_name):
    deep_copied_model = model_copy(model, 'datagen_' + mut_name)
    if mut_name == 'whi': # ZCA white
        datagen = ImageDataGenerator(zca_whitening=True)
    elif mut_name == 'rot': # rotation
        datagen = ImageDataGenerator(rotation_range=45)
    elif mut_name == 'sh': # shear
        datagen = ImageDataGenerator(shear_range=35)
    elif mut_name == 'fl': # flip
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=False)
    # elif mut_name == ''

    train_datas, train_labels = train_dataset

    # select a portion of data and reproduce
    number_of_train_data = len(train_datas)
    number_of_aug = math.floor(number_of_train_data * mutation_ratio)
    print("augment modified %d training datasets, totally %d" % (number_of_aug, number_of_train_data))

    aug_train_datas = train_datas[:number_of_aug]
    aug_train_labels = train_labels[:number_of_aug]
    # fit parameters from data
    datagen.fit(aug_train_datas)
    # configure batch size and retrieve one batch of images
    for datas_batch, labels_batch in datagen.flow(aug_train_datas, aug_train_labels, batch_size=number_of_aug):
        break # just gain the batches
    return (datas_batch, labels_batch), deep_copied_model

# 1. Data repetition: mutate_ratio, small impact on acc.
def DR_mut(train_dataset, model, mutation_ratio):
    deep_copied_model = model_copy(model, 'DR')
    train_datas, train_labels = train_dataset
    

    # shuffle the original train data
    shuffled_train_datas, shuffled_train_labels = shuffle_in_uni(train_datas, train_labels)

    # select a portion of data and reproduce
    number_of_train_data = len(train_datas)
    number_of_duplicate = math.floor(number_of_train_data * mutation_ratio)
    print("ramdomly repeat %d training datasets, totally %d" % (number_of_duplicate, number_of_train_data + number_of_duplicate))

    repeated_train_datas = shuffled_train_datas[:number_of_duplicate]
    repeated_train_labels = shuffled_train_labels[:number_of_duplicate]
    repeated_train_datas = np.append(train_datas, repeated_train_datas, axis=0)
    repeated_train_labels = np.append(train_labels, repeated_train_labels, axis=0)
    return (repeated_train_datas, repeated_train_labels), deep_copied_model

# 2. LE (Label Error) : huge impact,  higher mutate_ratio, lower acc(even < 0.5);
def LE_mut(train_dataset, model, label_lower_bound, label_upper_bound, mutation_ratio):
    deep_copied_model = model_copy(model, 'LE')
    train_datas, train_labels = train_dataset
    LE_train_datas, LE_train_labels = train_datas.copy(), train_labels.copy()


    number_of_train_data = len(LE_train_datas)
    number_of_error_labels = math.floor(number_of_train_data * mutation_ratio)
    print("ramdomly falsify %d training datasets' labels: now %d correct labels" % (number_of_error_labels, number_of_train_data - number_of_error_labels))

    shuffled_indexes = np.random.permutation(number_of_train_data)
    shuffled_indexes = shuffled_indexes[:number_of_error_labels]
    for old_index, new_index in enumerate(shuffled_indexes):
        while True:
            val = random.randint(label_lower_bound, label_upper_bound)
            num_of_classes = label_upper_bound - label_lower_bound + 1
            val = keras.utils.np_utils.to_categorical(val, num_of_classes)
            if np.array_equal(LE_train_labels[new_index], val):
                continue
            else:
                LE_train_labels[new_index] = val
                break
    return (LE_train_datas, LE_train_labels), deep_copied_model

# 3. DM data missing
def DM_mut(train_dataset, model, mutation_ratio):
    deep_copied_model = model_copy(model, 'DM')
    train_datas, train_labels = train_dataset
    DM_train_datas, DM_train_labels = train_datas.copy(), train_labels.copy()

    number_of_train_data = len(DM_train_datas)
    number_of_deletion = math.floor(number_of_train_data * mutation_ratio)
    print("ramdomly delete %d training datasets: now %d datasets" % (number_of_deletion, number_of_train_data - number_of_deletion))

    shuffled_indexes = np.random.permutation(number_of_train_data)
    shuffled_indexes = shuffled_indexes[:number_of_deletion]


    # delete the selected data
    DM_train_datas = np.delete(DM_train_datas, shuffled_indexes, 0)
    DM_train_labels = np.delete(DM_train_labels, shuffled_indexes, 0)
    return (DM_train_datas, DM_train_labels), deep_copied_model

# 4. DF (Data Shuffle), small impact on acc.
def DF_mut(train_dataset, model, mutation_ratio):
    deep_copied_model = model_copy(model, 'DF')
    train_datas, train_labels = train_dataset
    DF_train_datas, DF_train_labels = train_datas.copy(), train_labels.copy()


    number_of_train_data = len(DF_train_datas)
    number_of_shuffled_datas = math.floor(number_of_train_data * mutation_ratio)
    print("shuffle %d training datasets/%d datasets totally" % (number_of_shuffled_datas, number_of_train_data))

    shuffled_indexes = np.random.permutation(number_of_train_data)
    shuffled_indexes = shuffled_indexes[:number_of_shuffled_datas]

    DF_train_datas, DF_train_labels = shuffle_in_uni_with_permutation(DF_train_datas, DF_train_labels, shuffled_indexes)
    return (DF_train_datas, DF_train_labels), deep_copied_model

# 5.NP Noise Perturb: higher mutate_ratio, lower acc; higher STD, higher random_noise , lower acc
def NP_mut(train_dataset, model, mutation_ratio, STD=0.1):
    deep_copied_model = model_copy(model, 'NP')
    train_datas, train_labels = train_dataset
    NP_train_datas, NP_train_labels = train_datas.copy(), train_labels.copy()


    number_of_train_data = len(NP_train_datas)
    number_of_noise_perturbs = math.floor(number_of_train_data * mutation_ratio) # a specific amount of samples is chosen
    print("noise perturb %d training datasets/%d datasets totally with STD %f" % (number_of_noise_perturbs, number_of_train_data, STD))

    shuffled_indexes = np.random.permutation(number_of_train_data) # a list of INDEX!! Parameters: int or array_like If x is an integer, randomly permute np.arange(x).
    shuffled_indexes_to_add_noise = shuffled_indexes[:number_of_noise_perturbs] # a list of index!!! 

    
    random_noise = np.random.standard_normal(NP_train_datas.shape) * STD # Draw samples from a standard Normal distribution (mean=0, stdev=1)
    for new_index in shuffled_indexes_to_add_noise: # new_index, chosen sample's index
        # NP_train_datas[new_index] is the chosen image，random_noise[new_index]
        NP_train_datas[new_index] += random_noise[new_index]
    return (NP_train_datas, NP_train_labels), deep_copied_model

# 6. The AFRs operator randomly removes all the activation functions of a layer,
# to mimic the situation that the developer forgets to add the activation layers.
def AFRs_mut(train_dataset, model, mutated_layer_indices=None):
    # Copying and some assertions
    deep_copied_model = model_copy(model, 'AFRs')
    train_datas, train_labels = train_dataset
    copied_train_datas, copied_train_labels = train_datas.copy(), train_labels.copy()


    # Randomly select from suitable layers instead of the first one
    index_of_suitable_layers = []
    layers = [l for l in model.layers]
    for index, layer in enumerate(layers):
        if index == (len(model.layers) - 1):
            continue
        try:
            if layer.activation is not None:
                index_of_suitable_layers.append(index)
        except:
            pass
    number_of_suitable_layers = len(index_of_suitable_layers)
    if number_of_suitable_layers == 0:
        print('None activation of layers be removed, there is no suitable layer for the input model')
        return (copied_train_datas, copied_train_labels), deep_copied_model

    new_model = keras.models.Sequential()
    layers = [l for l in deep_copied_model.layers]

    if mutated_layer_indices == None:
        random_picked_layer_index = index_of_suitable_layers[random.randint(0, number_of_suitable_layers-1)]
        print('AFRs mutation operator ramdomly change layer %d\'s activation function into a linear f(x) = x' % random_picked_layer_index)

        for index, layer in enumerate(layers):
            if index == random_picked_layer_index:
                layer.activation = lambda x: x  # change into a linear function f(x) = x
                new_model.add(layer)
                continue
            new_model.add(layer)
    else:
        if mutated_layer_indices is not None:
            for index in mutated_layer_indices:
                assert index in index_of_suitable_layers, 'Index ' + str(index) + ' is an invalid index for this mutation'
                pass
        for index, layer in enumerate(layers):
            if index in mutated_layer_indices:
                layer.activation = lambda x: x  # change into a linear function f(x) = x
                new_model.add(layer)
                continue
            new_model.add(layer)

    return (copied_train_datas, copied_train_labels), new_model

def shuffle(a):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    length = len(a)
    permutation = np.random.permutation(length)
    index_permutation = np.arange(length)
    shuffled_a[permutation] = a[index_permutation]
    return shuffled_a

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

def shuffle_in_uni_with_permutation(a, b, permutation):
    assert len(a) == len(b)
    shuffled_a, shuffled_b = a.copy(), b.copy()
    shuffled_permutation = shuffle(permutation)
    shuffled_a[shuffled_permutation] = a[permutation]
    shuffled_b[shuffled_permutation] = b[permutation]
    return shuffled_a, shuffled_b

def model_copy(model, mode=''):
    original_layers = [l for l in model.layers]
    suffix = '_copy_' + mode
    new_model = keras.models.clone_model(model)
    for index, layer in enumerate(new_model.layers):
        original_layer = original_layers[index]
        original_weights = original_layer.get_weights()
        layer.name = layer.name + suffix
        layer.set_weights(original_weights)
    new_model.name = new_model.name + suffix
    return new_model