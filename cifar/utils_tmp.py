# -*- coding: utf-8 -*-

import numpy as np
import random
import os
from cv2 import imwrite
from collections import defaultdict
from datetime import datetime
from keras.models import *
from keras import backend as K # 使用抽象 Keras 后端编写新代码


import utils, network, source_mut_operators
# utils = utils.GeneralUtils()
network = network.CNNNetwork() #
source_mut_opts = source_mut_operators.SourceMutationOperators()

def load_file(file):
    if os.path.exists(file):
        print('%s loaded' % file)
        return np.load(file).item()
    else:
        print('No %s exists' % file)
        os._exit(0)

total_test_cases_num = 0
def run_operator(mutation_ratios, operator_name, model, train_dataset, \
test_datas, test_labels, seeds_dir, AFRs_mutated_layer_indices = None):
    # constant
    img_height, img_width, img_channels_num = 32, 32 ,3

    global total_test_cases_num # will be modified
    # local var
    (train_datas, train_labels) = train_dataset
    gen_test_cases = np.zeros(train_datas.shape) # train_datas 是 global var
    gen_test_cases = np.delete(gen_test_cases, slice(0, gen_test_cases.shape[0]), axis=0) # remove the all the lines, only keeps the shape (0, 32, 32, 1)
    difference_indexes = [] # a set for all differences among different operators
    counter = 0

    right_labels = np.argmax(test_labels, axis=1)  # right answers
    print("\n-----------------------------" + operator_name + " mutation operator-----------------------------")
    for mutation_ratio in mutation_ratios:
        if operator_name == 'DR':
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DR_mut(train_dataset, model, mutation_ratio)
        elif operator_name == 'LE':
            lower_bound = 0
            upper_bound = 9
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LE_mut(train_dataset, model, lower_bound, upper_bound, mutation_ratio)
        elif operator_name == 'DM':
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DM_mut(train_dataset, model, mutation_ratio)
        elif operator_name == 'DF':
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DF_mut(train_dataset, model, mutation_ratio)
        elif operator_name == 'NP':
            STD = 5
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.NP_mut(train_dataset, model, mutation_ratio, STD=STD)
        elif operator_name == 'LR':
            mutated_layer_indices = None
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LR_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif operator_name == 'LA':
            mutated_layer_indices = None
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LAs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif operator_name == 'AFR':
            mutated_layer_indices = AFRs_mutated_layer_indices
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.AFRs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif operator_name == 'whi' or operator_name == 'rot' or operator_name == 'sh' or operator_name == 'fl':
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.aug_mut(train_dataset, model, mutation_ratio, operator_name)
        else:
            print("Input is not a valid operator mode")
            return

        # compile model
        trained_model = network.compile_model(model)
        mutated_model = network.compile_model(mutated_model)

        trained_model = network.train_model(model, train_datas, train_labels)
        trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_labels)

        # ---------------------------------find different behaviors---------------------------------
        # quality control of mutant model:  evaluate model and get accurracy
        trained_loss, trained_acc = trained_model.evaluate(test_datas, test_labels, verbose=False)
        mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
        if mutant_acc < trained_acc - 0.05 * mutation_ratio:
          print("\n{0}th bad mutant with low acc {1:.2%} < {2:.2%}, mutation ratio {3:.3f}, will be dropped out\n".format(counter + 1, \
          mutant_acc, trained_acc - 0.05 * mutation_ratio, mutation_ratio))
          continue
        print("\n{0}th mutant passes the quality test, with acc {1:.2%} >= {2:.2%} and mutation ratio {3:.3f}".format(counter + 1, \
        mutant_acc, trained_acc - 0.05 * (counter + 1), mutation_ratio))

        # get the min and max of each neuron, store in a dict


        # test the mutated model

        origit_predi_labels = np.argmax(trained_model.predict(test_datas), axis = 1)
        mutant_predi_labels = np.argmax(trained_mutated_model.predict(test_datas), axis = 1)

        # compare the test reasults with correct result
        #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
        difference_indexes = difference_indexes + list(np.nonzero(origit_predi_labels - mutant_predi_labels)[0])
        difference_score = len(difference_indexes) / origit_predi_labels.size

        prev_gen_test_cases = gen_test_cases
        additional_test_cases = test_datas[np.nonzero(origit_predi_labels - mutant_predi_labels)[0]]
        concat_test_cases = np.append(gen_test_cases, additional_test_cases,axis = 0)
        _, idx = np.unique(concat_test_cases, axis = 0, return_index=True)
        gen_test_cases = concat_test_cases[np.sort(idx)]

        counter += 1
        print("New test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))
        # end of loop

    # save the test cases causing differences
    store_test_cases(operator_name, difference_indexes, right_labels, gen_test_cases, seeds_dir,
                     img_height, img_width, img_channels_num)
    total_test_cases_num += gen_test_cases.shape[0]
    print("now %d test cases are saved\n" % total_test_cases_num)

# =======================================For other coverage metrics=============================================================

# dict.value： multiple values in a list, 0 / 1+
def init_multisection_coverage_value(model, multisection_num):
    neurons_multisection_coverage_values = defaultdict(float)
    for layer in model.layers:
        # 对于不经过activation的layer, 不考虑其coverage
        if 'flatten' in layer.name or 'input' in layer.name  or 'dropout' in layer.name:
            continue
        # 对于经过activation的layer
        for index in range(layer.output_shape[-1]): # 输出张量 last D

            neurons_multisection_coverage_values[(layer.name, index)] = [0] * multisection_num # [0,0,0,....]
    return neurons_multisection_coverage_values


def update_multi_coverages(input_data, model, model_layer_times, model_neuron_values, multisection_coverage, multisection_num,\
upper_corner_coverage, lower_corner_coverage, threshold = 0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name and 'dropout' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])# list 嵌套 每层的output（list）

    # 因为是list，总的predict会broadcast到每个element(layer)->每层单独的intermediate output存在list中！！！
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data) # layers num: Lenet1 6, lenet4 7, lenent5 8 即选定layer层数

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):

        scaled = scale(intermediate_layer_output[0])

        for no_neuron in range(scaled.shape[-1]):
            # ... 相当于 :, 每一个filter/neuron 的结果, 等价于 scaled[:,:,no_neuron].
            # no_neuron 是卷积/maxpooling得到的一层neuron 直接结果: 对feature map 取mean
            new_output = np.mean(scaled[..., no_neuron])
        # coverage metrics for each neuron:
            # 1. baisc NC
            if new_output > threshold: # and model_layer_dict[(layer_names[i], no_neuron)] == 0:
                model_layer_times[(layer_names[i], no_neuron)] += 1 # 记作 该nueron被cover

            # 2. multisection_coverage
            upper_bound = model_neuron_values[(layer_names[i], no_neuron)][0] # max gained from training data
            lower_bound = model_neuron_values[(layer_names[i], no_neuron)][1] # min gained from training data, for CNN with Relu, this is 0
            sections = np.linspace(lower_bound, upper_bound, multisection_num + 1) #  Create an array of X values evenly spaced between min and max
            # index ∈ [0, multisection_num] 所以不会out of index
            for index in range(multisection_num):
                if new_output > sections[index] and new_output <= sections[index+1]: # (strip, 2*strip]
                    multisection_coverage[(layer_names[i], no_neuron)][index] += 1
                    break
            # 3. upper_corner_coverage, same as baisc NC
            if new_output > upper_bound:
                upper_corner_coverage[(layer_names[i], no_neuron)] += 1
            if new_output < lower_bound:
                lower_corner_coverage[(layer_names[i], no_neuron)] += 1 # for CNN with Relu, this is always  0

    return intermediate_layer_outputs # dict for coverage



def update_coverage_value(input_data, model, model_layer_value):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name and 'dropout' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # range(scaled.shape[-1])
        for num_neuron in range(scaled.shape[-1]):
            model_layer_value[(layer_names[i], num_neuron)] = np.mean(scaled[..., num_neuron])

    return intermediate_layer_outputs

# ====================================================================================================


def update_basic_coverage(input_data, model, model_layer_times, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name and 'dropout' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # range(scaled.shape[-1])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold: #and model_layer_dict[(layer_names[i], num_neuron)] == 0:
                model_layer_times[(layer_names[i], num_neuron)] += 1

    return intermediate_layer_outputs



# for SINGLE input data
def update_neuron_value(input_data, model, model_neuron_values):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name and 'dropout' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # range(scaled.shape[-1])
        for num_neuron in range(scaled.shape[-1]):
            new_output = np.mean(scaled[..., num_neuron])
            prev_max = model_neuron_values[(layer_names[i], num_neuron)][0]
            prev_min = model_neuron_values[(layer_names[i], num_neuron)][1]
            if new_output > prev_max : # max
                model_neuron_values[(layer_names[i], num_neuron)][0] = new_output
            elif new_output < prev_min: # min
                model_neuron_values[(layer_names[i], num_neuron)][1] = new_output
            else:
                pass
    return intermediate_layer_outputs



def shuffle_in_uni(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    length = len(a)
    permutation = np.random.permutation(length) # random
    index_permutation = np.arange(length)
    shuffled_a[permutation] = a[index_permutation]
    shuffled_b[permutation] = b[index_permutation]
    return shuffled_a, shuffled_b


# -------------------------------------------------------------------

from keras.preprocessing import image
def preprocess_image(img_path, target_h, target_w, target_channels_num):
    img = image.load_img(img_path, target_size=(target_h, target_w))
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, target_h, target_w, target_channels_num)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    return input_img_data

def deprocess_image(x):
    # de-normalization: [0,1] -> [0,255]
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2], x.shape[3])

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# -------------------------------------------------------------------------------
def remove_duplicate_without_sorting(seq):
   # order preserving
   noDupes = []
   [noDupes.append(i) for i in seq if not noDupes.count(i)]
   return noDupes

def clear_store_orig_imgs(test_cases, labels, save_dir):
    init_storage_dir(save_dir)
    for index, test_case in enumerate(test_cases):
        imwrite('%s%d_%d.png' % (save_dir, index + 1, labels[index]), test_case.reshape(32,32,3))

def store_test_cases(mutation_name, difference_indexes, right_labels, gen_test_cases, save_dir, \
                     target_h, target_w, target_channels_num):
    difference_indexes = remove_duplicate_without_sorting(difference_indexes)
    for index, test_case in enumerate(gen_test_cases):
        imwrite('%s%d_%d.png' % (save_dir + mutation_name, index + 1,\
         right_labels[difference_indexes[index]]), deprocess_image(test_case.reshape(1,target_h, target_w, target_channels_num)))
# ------------------------------------------------------------

def init_storage_dir(save_dir):
    if os.path.exists(save_dir):
        for i in os.listdir(save_dir):
            path_file = os.path.join(save_dir, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
    # if storage dir not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def init_coverage_times(model):
    model_layer_times = defaultdict(int)
    init_times(model,model_layer_times)
    return model_layer_times

def init_coverage_value(model):
    model_layer_value = defaultdict(float)
    init_times(model, model_layer_value)
    return model_layer_value

# dict.value：single value, 0 / 1+
def init_neuron_coverage(model, model_layer_times): # 将model的一些'cover有效的'layers的model_layer_times 整个由一个dict 来表示
    for layer in model.layers:
        # 对于不经过activation的layer, 不考虑其coverage
        if 'flatten' in layer.name or 'input' in layer.name or 'dropout' in layer.name:
            continue
        # 对于经过activation的layer
        for index in range(layer.output_shape[-1]): # 输出张量 last D
            model_layer_times[(layer.name, index)] = 0 # 'l层 第n个 Neuron' 整个tuple(layer.name, index) 做key代表 Neuron

def init_times(model,model_layer_times):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name or 'dropout' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_times[(layer.name, index)] = 0

def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_to_cover(not_covered,model_layer_dict):
    if not_covered:
        layer_name, index = random.choice(not_covered)
        not_covered.remove((layer_name, index))
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index

def random_strategy(model,model_layer_times, neuron_to_cover_num):
    loss_neuron = []
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_times.items() if v == 0]
    for _ in range(neuron_to_cover_num):
        layer_name, index = neuron_to_cover(not_covered, model_layer_times)
        loss00_neuron = K.mean(model.get_layer(layer_name).output[..., index])
        # if loss_neuron == 0:
        #     loss_neuron = loss00_neuron
        # else:
        #     loss_neuron += loss00_neuron
        # loss_neuron += loss1_neuron
        loss_neuron.append(loss00_neuron)
    return loss_neuron


model_layer_weights_top_k = []
def neuron_select_high_weight(model, layer_names, top_k):
    global model_layer_weights_top_k
    model_layer_weights_dict = {}
    for layer_name in layer_names:
        weights = model.get_layer(layer_name).get_weights()
        if len(weights) <= 0:
            continue
        w = np.asarray(weights[0])  # 0 is weights, 1 is biases
        w = w.reshape(w.shape)
        for index in range(model.get_layer(layer_name).output_shape[-1]):
            index_w = np.mean(w[..., index])
            if index_w <= 0:
                continue
            model_layer_weights_dict[(layer_name,index)]=index_w
    # notice!
    model_layer_weights_list = sorted(model_layer_weights_dict.items(), key=lambda x: x[1], reverse=True)

    k = 0
    for (layer_name, index),weight in model_layer_weights_list:
        if k >= top_k:
            break
        model_layer_weights_top_k.append([layer_name,index])
        k += 1


def target_neurons_in_grad(model, model_layer_times, model_layer_value, neuron_select_strategy, neuron_to_cover_num, threshold):
    if neuron_select_strategy == 'None':
        return random_strategy(model, model_layer_times, neuron_to_cover_num)

    num_strategy = len([x for x in neuron_select_strategy if x in ['1', '2', '3', '4']])
    neuron_to_cover_num_each = neuron_to_cover_num // num_strategy

    loss_neuron = []
    # initialization for strategies
    if ('1' in list(neuron_select_strategy)) or ('2' in list(neuron_select_strategy)):
        i = 0
        neurons_covered_times = []
        neurons_key_pos = {}
        for (layer_name, index), time in model_layer_times.items():
            neurons_covered_times.append(time)
            neurons_key_pos[i] = (layer_name, index)
            i += 1
        neurons_covered_times = np.asarray(neurons_covered_times)
        times_total = sum(neurons_covered_times)

    # select neurons covered often
    if '1' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)#The beginning of no neurons covered
        neurons_covered_percentage = neurons_covered_times / float(times_total)
        # num_neuron0 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage)
        num_neuron0 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False, p=neurons_covered_percentage)
        for num in num_neuron0:
            layer_name0, index0 = neurons_key_pos[num]
            loss0_neuron = K.mean(model.get_layer(layer_name0).output[..., index0])
            loss_neuron.append(loss0_neuron)

    # select neurons covered rarely
    if '2' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)
        neurons_covered_times_inverse = np.subtract(max(neurons_covered_times), neurons_covered_times)
        neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(sum(neurons_covered_times_inverse))
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
        num_neuron1 = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_each, replace=False,
                                       p=neurons_covered_percentage_inverse)
        for num in num_neuron1:
            layer_name1, index1 = neurons_key_pos[num]
            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
            loss_neuron.append(loss1_neuron)

    # select neurons with largest weights (feature maps with largest filter weights)
    if '3' in list(neuron_select_strategy):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name and 'dropout' not in layer.name]
        k = 0.1
        top_k = k * len(model_layer_times)  # number of neurons to be selected within
        global model_layer_weights_top_k
        if len(model_layer_weights_top_k) == 0:
            neuron_select_high_weight(model, layer_names, top_k)  # Set the value

        num_neuron2 = np.random.choice(range(len(model_layer_weights_top_k)), neuron_to_cover_num_each, replace=False)
        for i in num_neuron2:
            # i = np.random.choice(range(len(model_layer_weights_top_k)))
            layer_name2 = model_layer_weights_top_k[i][0]
            index2 = model_layer_weights_top_k[i][1]
            loss2_neuron = K.mean(model.get_layer(layer_name2).output[..., index2])
            loss_neuron.append(loss2_neuron)

    if '4' in list(neuron_select_strategy):
        above_threshold = []
        below_threshold = []
        above_num = neuron_to_cover_num_each / 2
        below_num = neuron_to_cover_num_each - above_num
        above_i = 0
        below_i = 0
        for (layer_name, index), value in model_layer_value.items():
            if threshold + 0.25 > value > threshold and 'dense' not in layer_name and \
                    'activation' not in layer_name and layer_name != 'before_softmax' \
                    and above_i < above_num:
                above_threshold.append([layer_name, index])
                above_i += 1
                # print(layer_name,index,value)
                # above_threshold_dict[(layer_name, index)]=value
            elif threshold > value > threshold - 0.2 and 'dense' not in layer_name and \
                    'activation' not in layer_name and layer_name != 'before_softmax' \
                    and below_i < below_num:
                below_threshold.append([layer_name, index])
                below_i += 1

        loss_neuron = []
        if len(above_threshold) > 0:
            for above_item in range(len(above_threshold)):
                loss_neuron.append(K.mean(
                    model.get_layer(above_threshold[above_item][0]).output[..., above_threshold[above_item][1]]))

        if len(below_threshold) > 0:
            for below_item in range(len(below_threshold)):
                loss_neuron.append(-K.mean(
                    model.get_layer(below_threshold[below_item][0]).output[..., below_threshold[below_item][1]]))

        if loss_neuron == 0:
            return random_strategy(model, model_layer_times, 1)  # The beginning of no neurons covered

    return loss_neuron



def neuron_covered(model_layer_times):
    covered_neurons = len([v for v in model_layer_times.values() if v > 0])
    total_neurons = len(model_layer_times)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def get_signature():
    now = datetime.now()
    past = datetime(2015, 6, 6, 0, 0, 0, 0)
    timespan = now - past
    time_sig = int(timespan.total_seconds() * 1000)

    return str(time_sig)
