# -*- coding: utf-8 -*-
import numpy as np
import keras

import random, math
import unittest

class GeneralUtils():

    def __init__(self):
        pass

    '''
    Return True with prob
    Input: probability within [0, 1]
    Ouput: True or False
    '''
    def decision(self, prob):
        assert prob >= 0, 'Probability should in the range of [0, 1]'
        assert prob <= 1, 'Probability should in the range of [0, 1]'
        return random.random() < prob

    def generate_permutation(self, size_of_permutation, extract_portion):
        assert extract_portion <= 1
        num_of_extraction = math.floor(size_of_permutation * extract_portion)
        permutation = np.random.permutation(size_of_permutation)
        permutation = permutation[:num_of_extraction]
        return permutation

    def shuffle(self, a):
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        length = len(a)
        permutation = np.random.permutation(length)
        index_permutation = np.arange(length)
        shuffled_a[permutation] = a[index_permutation]
        return shuffled_a

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

    def shuffle_in_uni_with_permutation(self, a, b, permutation):
        assert len(a) == len(b)
        shuffled_a, shuffled_b = a.copy(), b.copy()
        shuffled_permutation = self.shuffle(permutation)
        shuffled_a[shuffled_permutation] = a[permutation]
        shuffled_b[shuffled_permutation] = b[permutation]
        return shuffled_a, shuffled_b

    '''
    SMM stands for source-level mutated model
    This function looks quite terrible and messy, should be simplified
    '''
    def print_messages_SMO(self, mode, train_datas=None, train_labels=None, mutated_datas=None, mutated_labels=None, model=None, mutated_model=None, mutation_ratio=0):
        if mode in ['DR', 'DM']:
            print('Before ' + mode)
            print('Train data shape:', train_datas.shape)
            print('Train labels shape:', train_labels.shape)
            print('')

            print('After ' + mode + ', where the mutation ratio is', mutation_ratio)
            print('Train data shape:', mutated_datas.shape)
            print('Train labels shape:', mutated_labels.shape)
            print('')
        elif mode in ['LE', 'DF', 'NP']:
            pass
        elif mode in ['LR', 'LAs', 'AFRs']:
            print('Original untrained model architecture:')
            model.summary()
            print('')

            print('Mutated untrained model architecture:')
            mutated_model.summary()
            print('')
        else:
            pass

    '''
    MMM stands for model-level mutated model
    '''
    def print_messages_MMM_generators(self, mode, network=None, test_datas=None, test_labels=None, model=None, mutated_model=None, STD=0.1, mutation_ratio=0):
        if mode in ['GF', 'WS', 'NEB', 'NAI', 'NS']:
            print('Before ' + mode)
            network.evaluate_model(model, test_datas, test_labels)
            print('After ' + mode + ', where the mutation ratio is', mutation_ratio)
            network.evaluate_model(mutated_model, test_datas, test_labels, mode)
        elif mode in ['LD', 'LAm', 'AFRm']:
            print('Before ' + mode)
            model.summary()
            network.evaluate_model(model, test_datas, test_labels)

            print('After ' + mode)
            mutated_model.summary()
            network.evaluate_model(mutated_model, test_datas, test_labels, mode)
        else:
            pass


class ModelUtils():

    def __init__(self):
        pass

    def print_layer_info(self, layer):
        layer_config = layer.get_config()
        print('Print layer configuration information:')
        for key, value in layer_config.items():
            print(key, value)

    def model_copy(self, model, mode=''):
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

    def get_booleans_of_layers_should_be_mutated(self, num_of_layers, indices):
        if indices == None:
            booleans_for_layers = np.full(num_of_layers, True)
        else:
            booleans_for_layers = np.full(num_of_layers, False)
            for index in indices:
                booleans_for_layers[index] = True
        return booleans_for_layers

    def print_comparision_of_layer_weights(self, old_model, new_model):
        old_layers = [l for l in old_model.layers]
        new_layers = [l for l in new_model.layers]
        assert len(old_layers) == len(new_layers)
        num_of_layers = len(old_layers)
        booleans_for_layers = np.full(num_of_layers, True)
        names_for_layers = []
        for index in range(num_of_layers):
            old_layer, new_layer = old_layers[index], new_layers[index]
            names_for_layers.append(type(old_layer).__name__)
            old_layer_weights, new_layer_weights = old_layer.get_weights(), new_layer.get_weights()
            if len(old_layer_weights) == 0:
                continue

            is_equal_connections = np.array_equal(old_layer_weights[0], new_layer_weights[0])
            is_equal_biases = np.array_equal(old_layer_weights[1], new_layer_weights[1])
            is_equal = is_equal_connections and is_equal_biases
            if not is_equal:
                booleans_for_layers[index] = False

        print('Comparision of weights between original model and mutated model,')
        print('If the weights of specific layer is modified, return True. Otherwise, return False')
        print('')
        print(' Layer index |   Layer name   | Is mutated ?')
        print(' -------------------------------------------')
        for index, result in enumerate(booleans_for_layers):
            name = names_for_layers[index]
            print(' {index} | {name} | {result}'.format(index=str(index).rjust(11), name=name.rjust(14), result=(not result)))
        print('')


class ExaminationalUtils():

    def __init__(self):
        pass

    def mutation_ratio_range_check(self, mutation_ratio):
        assert mutation_ratio >= 0, 'Mutation ratio attribute should in the range [0, 1]'
        assert mutation_ratio <= 1, 'Mutation ratio attribute should in the range [0, 1]'
        pass

    def training_dataset_consistent_length_check(self, lst_a, lst_b):
        assert len(lst_a) == len(lst_b), 'Training datas and labels should have the same length'
        pass

    def valid_indices_of_mutated_layers_check(self, num_of_layers, indices):
        if indices is not None:
            for index in indices:
                assert index >= 0, 'Index should be positive'
                assert index < num_of_layers, 'Index should not be out of range, where index should be smaller than ' + str(num_of_layers)
                pass

    def in_valid_indices_check(self, suitable_indices, indices):
        if indices is not None:
            for index in indices:
                assert index in suitable_indices, 'Index ' + str(index) + ' is an invalid index for this mutation'
                pass


import random
import os
from cv2 import imwrite
from collections import defaultdict
import numpy as np
from PIL import Image
from datetime import datetime
from keras import backend as K # 使用抽象 Keras 后端编写新代码
# 如果你希望你编写的 Keras 模块与 Theano (th) 和 TensorFlow (tf) 兼容，则必须通过抽象 Keras 后端 API 来编写它们。
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model


model_layer_weights_top_k = []

# -------------------------------------------------------------------------------------------------------------------


# def run_operator(total_test_cases_num, model_layer_values, mutation_ratios, mutant_accuracy_control_threshold, \
# operator_name, model, train_dataset, test_datas, test_labels, save_dir, AFRs_mutated_layer_indices = None):
#
#     # local var
#     # normal_accs = []
#     # mutant_accs = []
#     gen_test_cases = np.zeros(train_dataset[0].shape) # train_datas 是 global var
#     gen_test_cases = np.delete(gen_test_cases, slice(0, gen_test_cases.shape[0]), axis=0) # remove the all the lines, only keeps the shape (0, 32, 32, 1)
#     difference_indexes = [] # a set for indexes with differences
#     counter = 0
#
#     right_labels = np.argmax(test_labels, axis=1) # right answers
#
#     print("\n-----------------------------" + operator_name + " mutation operator-----------------------------")
#     for mutation_ratio in mutation_ratios:
#         if operator_name == 'DR':
#             (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DR_mut(train_dataset, model, mutation_ratio)
#         elif operator_name == 'LE':
#             lower_bound = 0
#             upper_bound = 9
#             (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LE_mut(train_dataset, model, lower_bound, upper_bound, mutation_ratio)
#         elif operator_name == 'DM':
#             (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DM_mut(train_dataset, model, mutation_ratio)
#         elif operator_name == 'DF':
#             (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DF_mut(train_dataset, model, mutation_ratio)
#         elif operator_name == 'NP':
#             STD = 5
#             (mutated_datas, mutated_labels), mutated_model = source_mut_opts.NP_mut(train_dataset, model, mutation_ratio, STD=STD)
#         elif operator_name == 'whi' or operator_name == 'rot' or operator_name == 'sh' or operator_name == 'fl':
#             (mutated_datas, mutated_labels), mutated_model = source_mut_opts.aug_mut(train_dataset, model, mutation_ratio, operator_name)
#
#         # elif operator_name == 'LR':
#         #     mutated_layer_indices = None
#         #     (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LR_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
#         # elif operator_name == 'LAs':
#         #     mutated_layer_indices = None
#         #     (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LAs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
#         elif operator_name == 'AFRs':
#             mutated_layer_indices = AFRs_mutated_layer_indices
#             (mutated_datas, mutated_labels), mutated_model = source_mut_opts.AFRs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
#         else:
#             print("Input is not a valid operator mode")
#             return
#
#         # compile model
#         # model = network.compile_model(model)
#         mutated_model = network.compile_model(mutated_model)
#
#         # train model
#         # trained_model = network.train_model(model, train_datas, train_labels)
#         trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_labels)
#
#         # evaluate model and get accurracy
#         # loss, acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
#         # normal_accs.append(acc)
#         # mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
#         # mutant_accs.append(mutant_acc)
#
#         # get the min and max of each neuron in original model! store in a dict
#     # if use_other_metric:
#         print("computing neuron value range...")
#         for i, single_training_example in enumerate(train_dataset[0]): # 默认遍历第一D
#             if i % 500 == 0:
#                 print("%d, " %i, end = "")
#             update_neuron_value(single_training_example.reshape(1, 28, 28, 1), model, model_layer_values)
#         print('Done.')
#
#         # ---------------------------------find different behaviors---------------------------------
#         # quality control of mutant model
#         mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
#         if mutant_acc < mutant_accuracy_control_threshold[counter]:
#           counter += 1
#           print("\n{0}th bad mutant with low acc {1:.2%} < {2:.2%}, mutation ratio {3:.3f}, will be dropped out\n".format(counter, \
#           mutant_acc, mutant_accuracy_control_threshold[counter], mutation_ratio))
#           continue
#         print("\n{0}th mutant passes the quality test, with acc {1:.2%} > {2:.2%} and mutation ratio {3:.3f}".format(counter + 1, \
#         mutant_acc, mutant_accuracy_control_threshold[counter], mutation_ratio))
#
#         # test the mutated model
#         mutant_predi_labels = np.argmax(trained_mutated_model.predict(test_datas), axis = 1)
#
#         # compare the test reasults with correct result
#         #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
#         difference_indexes = difference_indexes + list(np.nonzero(right_labels-mutant_predi_labels)[0])
#
#         # collect all selected test cases
#         prev_gen_test_cases = gen_test_cases
#         additional_test_cases = test_datas[np.nonzero(right_labels - mutant_predi_labels)[0]]
#         concat_test_cases = np.append(gen_test_cases, additional_test_cases,axis = 0)
#         _, idx = np.unique(concat_test_cases, axis = 0, return_index=True)
#         gen_test_cases = concat_test_cases[np.sort(idx)]
#
#         counter += 1
#         print("New test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))
#         # end of loop
#
#     # save the test cases causing differences
#     store_test_cases(save_dir, operator_name, difference_indexes, right_labels, gen_test_cases)
#
#     total_test_cases_num += gen_test_cases.shape[0]
#     print("-----------------------now %d test cases are saved-----------------------\n" % total_test_cases_num)


