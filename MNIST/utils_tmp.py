# -*- coding: utf-8 -*-

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
total_test_cases_num = 0

def load_file(file):
    if os.path.exists(file):
        print("%s loaded" % file)
        return np.load(file).item()
    else:
        print('No %s exists' % file)
        os._exit(0)

from keras.preprocessing import image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data

def deprocess_image(x):
    # de-normalization: [0,1] -> [0,255]
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2]) 


def decode_label(pred):
    return decode_predictions(pred)[0][0][1]


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

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
        imwrite('%s%d_%d.png' % (save_dir + mutation_name, index + 1, \
        right_labels[difference_indexes[index]]), deprocess_image(test_case.reshape(1,28,28,1)))

# ------------------------------------------------------------------------
def get_heatmap(model, label, input_img, last_conv_name, last_conv_depth):
    # apply Grad-CAM algorithm to the generated adversrial examples and return heatmap
    advers_output = model.output[:, label]
    last_conv_layer = model.get_layer(last_conv_name)
    grads = K.gradients(advers_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_img])
    for conv_index in range(last_conv_depth):
        conv_layer_output_value[:, :, conv_index] *= pooled_grads_value[conv_index]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)        
    return heatmap

import cv2
def impose_heatmap_to_img(img_path, save_dir, heatmap, adversrial_no, label, advers_label):
    # We resize the heatmap to have the same size as the original image
    img = cv2.imread(img_path)                
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image            
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)                
    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img
    # Save the image to disk
    save_heatmap_name = save_dir + str(adversrial_no) + "_" + \
        str(label) + '_as_' + str(advers_label) + '_heat.png'
    imwrite(save_heatmap_name, superimposed_img)

# --------------------------------------------------
def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


# 使用dict时，如果引用的Key不存在，就会抛出KeyError。如果希望key不存在时，返回一个默认值，就可以用defaultdict
# 其他行为跟dict是完全一样的。
def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3

def init_coverage_tables(model1):
    model_layer_dict1 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1

# dict.value：single value, 0 / 1+
def init_neuron_coverage(model, model_layer_times): # 将model的一些'cover有效的'layers的model_layer_times 整个由一个dict 来表示
    for layer in model.layers:
        # 对于不经过activation的layer, 不考虑其coverage
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        # 对于经过activation的layer
        for index in range(layer.output_shape[-1]): # 输出张量 last D
            model_layer_times[(layer.name, index)] = 0 # 'l层 第n个 Neuron' 整个tuple(layer.name, index) 做key代表 Neuron

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
        # 对于不经过activation的layer, 不考虑其coverage
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        # 对于经过activation的layer
        for index in range(layer.output_shape[-1]): # 输出张量 last D
            # key: 'l层 第n个 Neuron' 整个tuple(layer.name, index) 代表 Neuron, value: (max, min)
            model_neuron_values[(layer.name, index)] = [0, 0]
    return model_neuron_values

# dict.value： multiple values in a list, 0 / 1+
def init_multisection_coverage_value(model, multisection_num):
    neurons_multisection_coverage_values = defaultdict(float)
    for layer in model.layers:
        # 对于不经过activation的layer, 不考虑其coverage
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        # 对于经过activation的layer
        for index in range(layer.output_shape[-1]): # 输出张量 last D

            neurons_multisection_coverage_values[(layer.name, index)] = [0] * multisection_num # [0,0,0,....]
    return neurons_multisection_coverage_values

def init_storage_dir(save_dir):
    if os.path.exists(save_dir):
        for i in os.listdir(save_dir):
            path_file = os.path.join(save_dir, i)
            if os.path.isfile(path_file):
                os.remove(path_file)

    # if storage dir not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



# -------------------------------------------------------------------------------------------------------------------

import utils, network, source_mut_operators
network = network.CNNNetwork()
source_mut_opts = source_mut_operators.SourceMutationOperators()

def run_operator(mutation_ratios, operator_name, model, train_dataset, test_datas, test_labels, save_dir, AFRs_mutated_layer_indices = None):
    
    global total_test_cases_num # will be modified
    # local var
    (train_datas, train_labels) = train_dataset
    gen_test_cases = np.zeros(train_dataset[0].shape) # train_datas 是 global var
    gen_test_cases = np.delete(gen_test_cases, slice(0, gen_test_cases.shape[0]), axis=0) # remove the all the lines, only keeps the shape (0, 32, 32, 1)
    difference_indexes = [] # a set for indexes with differences
    counter = 0

    right_labels = np.argmax(test_labels, axis=1) # right answers

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
        elif operator_name == 'whi' or operator_name == 'rot' or operator_name == 'sh' or operator_name == 'fl':
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.aug_mut(train_dataset, model, mutation_ratio, operator_name)
        # elif operator_name == 'LR':
        #     mutated_layer_indices = None
        #     (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LR_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        # elif operator_name == 'LAs':
        #     mutated_layer_indices = None
        #     (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LAs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        elif operator_name == 'AFRs':
            mutated_layer_indices = AFRs_mutated_layer_indices
            (mutated_datas, mutated_labels), mutated_model = source_mut_opts.AFRs_mut(train_dataset, model, mutated_layer_indices=mutated_layer_indices)
        else:
            print("Input is not a valid operator mode")
            return

        # compile model
        trained_model = network.compile_model(model)
        mutated_model = network.compile_model(mutated_model)

        # train model
        trained_model = network.train_model(model, train_datas, train_labels)
        trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_labels)

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

# ------------------------------------------------------------------------------------------------
def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_to_cover(not_covered,model_layer_dict):
    if not_covered: # 只要有传参，赋了值
        layer_name, index = random.choice(not_covered)
        not_covered.remove((layer_name, index))
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index

# not_covered neurons的list中 random select neuron
def random_strategy(model,model_layer_times, target_neuron_cover_num):
    target_neurons_outputs = []
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_times.items() if v == 0]
    for _ in range(target_neuron_cover_num):
        # random 选择 not covered的neurons, 并计算出其value
        layer_name, index = neuron_to_cover(not_covered, model_layer_times)
    # Note: 为什么要计算并返回其output activation value? 因为max neuron coverage 就是 max value 使其activate !!!
        loss00_neuron = K.mean(model.get_layer(layer_name).output[..., index]) # keras.backend.mean, 张量在某一指定轴的均值

        # if target_neurons_outputs == 0:
        #     target_neurons_outputs = loss00_neuron
        # else:
        #     target_neurons_outputs += loss00_neuron
        # target_neurons_outputs += low_covered_neuron_output
        target_neurons_outputs.append(loss00_neuron)
    return target_neurons_outputs

def neuron_select_high_weight(model, layer_names, top_k):
    global model_layer_weights_top_k
    model_layer_weights_dict = {} # 作为比较的中介，不是最后的返回值...
    for layer_name in layer_names:
        weights = model.get_layer(layer_name).get_weights() # 一个layer一起获取
        if len(weights) <= 0:
            continue # skip the layer
        w = np.asarray(weights[0])  # 0 is weights, 1 is biases
        w = w.reshape(w.shape) # 整个一层的, 是四维(f, f, n[l-1], n[l]), 共 neuron数量
        for index in range(model.get_layer(layer_name).output_shape[-1]): # layer.output_shape[-1] 就是neuron数量
            index_w = np.mean(w[..., index]) # neuron(filter) weight之和取 mean
            if index_w <= 0:
                continue
            model_layer_weights_dict[(layer_name,index)] = index_w
    # notice!
    model_layer_weights_list = sorted(model_layer_weights_dict.items(), key=lambda x: x[1], reverse=True)

    k = 0
    for (layer_name, index),weight in model_layer_weights_list:
        # 获取前k个 neuron， 是用global var记下的是他们的[layer_name,index]
        if k >= top_k: 
            break
        model_layer_weights_top_k.append([layer_name,index])
        k += 1


def target_neurons_in_grad(model, model_layer_times, model_layer_value, neuron_select_strategy, target_neuron_cover_num, threshold):
    if neuron_select_strategy == 'None':
        return random_strategy(model, model_layer_times, target_neuron_cover_num)

    # neuron_select_strategy input 允许使用多个strategy: neuron_select_strategy是str !!!
    num_strategy = len([x for x in neuron_select_strategy if x in ['1', '2', '3', '4']])
    target_neuron_cover_num_each = int(target_neuron_cover_num / num_strategy) # 将需要cover的平均给每个strategy

    target_neurons_outputs = []

    # initialization for strategies
    if ('1' in list(neuron_select_strategy)) or ('2' in list(neuron_select_strategy)):
        i = 0
        neurons_covered_times = []
        neurons_key_pos = {}
        for (layer_name, index), time in model_layer_times.items(): # get coverage so far
            #  (layer_name, index), time 有对应顺序
            neurons_covered_times.append(time) # list [neuron's coverage, neuron's coverage, ...]
            neurons_key_pos[i] = (layer_name, index) # list [neuron, neuron, ...]
            i += 1
        neurons_covered_times = np.asarray(neurons_covered_times) # list -> nd array
        total_neurons_covered_times = neurons_covered_times.sum() # sum(neurons_covered_times)
        # sum up an 1-D array, Python built-in sum is OK,as list does ; but for 2D 3D...array, use np.sum() is to sum up all

    # select neurons covered frequently during LAST testing.
    if '1' in list(neuron_select_strategy):
        if total_neurons_covered_times == 0: # no difference of coverage between neurons, so random pick, no care for strategies
            print("total_neurons_covered_times = 0, strategies fails")
            return random_strategy(model, model_layer_times, 1) # why 1 ? 因为neurons_covered_times= 0, strategy 不管用， 随机选一个
        # total_neurons_covered_times != 0
        neurons_covered_percentage = neurons_covered_times / float(total_neurons_covered_times) # neurons_covered_times(list) braodcasting,

        # 从len(neurons_covered_times))中随机选 target_neuron_cover_num_each个,
        # high_covered_neurons_indices存的是index !!! 从range的list中pick        
        # print(target_neuron_cover_num_each)
        high_covered_neurons_indices = np.random.choice(range(len(neurons_covered_times)), \
            target_neuron_cover_num_each, replace=False, # False: not put back after choosing
            p = neurons_covered_percentage) # higher neurons_covered_percentage, more likely to be picked

        for num in high_covered_neurons_indices:
            layer_name0, index0 = neurons_key_pos[num] # list [neuron, neuron, ...]
            high_covered_neuron_output = K.mean(model.get_layer(layer_name0).output[..., index0]) # get their previous output !!!
            target_neurons_outputs.append(high_covered_neuron_output) # list [output, output, ...], to be maximized

    # select neurons covered rarely
    if '2' in list(neuron_select_strategy):
        if total_neurons_covered_times == 0:
            print("total_neurons_covered_times = 0, strategies fails")
            return random_strategy(model, model_layer_times, 1)
        # 唯一和上面不一样之处: (np.subtract 是 element wise)，每个数都减去 max后都是负数了，越小的则说明之前time越小
        neurons_covered_times_inverse = np.subtract(max(neurons_covered_times), neurons_covered_times)
        # trick： 让越小的负数 被pick的prop越大， 就是全体除以同一个负数，这样又变正了
        neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(sum(neurons_covered_times_inverse))
        # low_covered_neurons_indices = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)        
        low_covered_neurons_indices = np.random.choice(range(len(neurons_covered_times)), \
                target_neuron_cover_num_each, replace=False,
                p = neurons_covered_percentage_inverse)
        for num in low_covered_neurons_indices:
            layer_name1, index1 = neurons_key_pos[num] # 锁定neuron
            low_covered_neuron_output = K.mean(model.get_layer(layer_name1).output[..., index1])
            target_neurons_outputs.append(low_covered_neuron_output)

    # select neurons with largest weights (feature maps with largest filter weights)， 
    # 作者 assumption that neurons with top weights maybe have larger in$uence on other neurons.
    # 结果观察：adversarial norm 较大 解释：可能 进一步放大了这些weight对应的feature 
    # 而本身'强力'(weight大)的neuron已经是cover较多了，反而使得adversrial难生成
    if '3' in list(neuron_select_strategy):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]
        k = 0.1
        top_k = k * len(model_layer_times)  # number of neurons to be selected within
        global model_layer_weights_top_k
        if len(model_layer_weights_top_k) == 0:
            neuron_select_high_weight(model, layer_names, top_k)  # Set the value

        large_weight_neurons_indices = np.random.choice(range(len(model_layer_weights_top_k)), target_neuron_cover_num_each, replace=False)
        for i in large_weight_neurons_indices:
            # i = np.random.choice(range(len(model_layer_weights_top_k)))
            layer_name2 = model_layer_weights_top_k[i][0]
            index2 = model_layer_weights_top_k[i][1]
            large_weight_neuron_output = K.mean(model.get_layer(layer_name2).output[..., index2])
            target_neurons_outputs.append(large_weight_neuron_output)

    # select neurons near the activation threshold.
    # It is easier to accelerate if activating/deactivating neurons with output value slightly lower/larger than the threshold.
    if '4' in list(neuron_select_strategy):
        above_threshold = []
        below_threshold = []
        above_num = target_neuron_cover_num_each / 2
        below_num = target_neuron_cover_num_each - above_num
        above_i = 0
        below_i = 0
        for (layer_name, index), value in model_layer_value.items():
            if threshold + 0.25 > value > threshold and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and above_i < above_num:
                above_threshold.append([layer_name, index])
                above_i += 1
                # print(layer_name,index,value)
                # above_threshold_dict[(layer_name, index)]=value
            elif threshold > value > threshold - 0.2 and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and below_i < below_num:
                below_threshold.append([layer_name, index])
                below_i += 1
        #
        # loss3_neuron_above = 0
        # loss3_neuron_below = 0
        target_neurons_outputs = []
        if len(above_threshold) > 0:
            for above_item in range(len(above_threshold)):
                target_neurons_outputs.append(K.mean(
                    model.get_layer(above_threshold[above_item][0]).output[..., above_threshold[above_item][1]]))

        if len(below_threshold) > 0:
            for below_item in range(len(below_threshold)):
                target_neurons_outputs.append(-K.mean(
                    model.get_layer(below_threshold[below_item][0]).output[..., below_threshold[below_item][1]]))

        if target_neurons_outputs == 0:
            print("total_neurons_covered_times = 0, strategies fails")
            return random_strategy(model, model_layer_times, 1)  # no satisfied neuron, random pick one

    return target_neurons_outputs # a list of outputs of certain neurons



# -------------------------------------------------------------------------
def neuron_scale(target_neurons_outputs):
    target_neurons_outputs_new = []
    loss_sum = K.sum(target_neurons_outputs)
    for loss_each in target_neurons_outputs:
        loss_each /= loss_sum
        target_neurons_outputs_new.append(loss_each)
    return target_neurons_outputs_new

def neuron_scale_maxmin(target_neurons_outputs):
    max_loss = K.max(target_neurons_outputs)
    min_loss = K.min(target_neurons_outputs)
    base = max_loss - min_loss
    target_neurons_outputs_new = []
    for loss_each in target_neurons_outputs:
        loss_each_new = (loss_each - min_loss) / base
        target_neurons_outputs_new.append(loss_each_new)
    return target_neurons_outputs_new

# access the dict of coverage times 一
def neuron_covered(model_layer_times):
    covered_neurons = len([v for v in model_layer_times.values() if v > 0])
    total_neurons = len(model_layer_times) # 一对k-v就是一个neuron
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

# 0-1 min-max scaling： 将数据按比例缩放，使之落入一个小的特定区间[0, 1]
def scale(intermediate_layer_output, rmax=1, rmin=0):
    # 做broadcasting
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin # 这里，x_scaled = X_std
    return X_scaled # broadcasting to the nd array


# 0-1 min-max scaling： 将数据按比例缩放，使之落入一个小的特定区间[0, 1]
def scale(intermediate_layer_output, rmax=1, rmin=0):
    # 做broadcasting
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin # 这里，x_scaled = X_std
    return X_scaled # broadcasting to the nd array

# for SINGLE input data
def update_neuron_value(input_data, model, model_neuron_values):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

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


def update_coverage(input_data, model, model_layer_times, model_neuron_values, multisection_coverage, multisection_num,\
upper_corner_coverage, lower_corner_coverage, threshold = 0):
    # Extract 已经训练好的 ！！the layers
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    # build the model with only the layers above， 使用Model class API形式
    intermediate_layer_model = Model(inputs=model.input, # model.input是原mnist traning data
                                     # get_layer根据名称（唯一）或索引值查找layer
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])# list 嵌套 每层的output（list）
    # print([model.get_layer(layer_name).output for layer_name in layer_names]) # length 6的list, 其element事故layer是nd array

    # 因为是list，总的predict会broadcast到每个element(layer)->每层单独的intermediate output存在list中！！！
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data) # layers num: Lenet1 6, lenet4 7, lenent5 8 即选定layer层数
    # debug
    # print(len(intermediate_layer_outputs))

    # 因为结果分别和threshold比较 判定covered与否
    #  intermediate_layer_output 第i层的output，是nd array(1, X, Y, f)。其中f 是layer的filter 即neuron的数量！！
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        # nd array[0] 等价于[0, :, :, :] 等于 取第一个sample: 每个element是在axis = 0的轴上选取第一个元素，同时选取axis = 1, 2,3 上的全部元素
        # 所以[0]等于抹去那个1 得到真正的 单个neuron output, (X,Y,f)的nd array
# Note： 统一scale 才能用同一个threshold 来判定activation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        scaled = scale(intermediate_layer_output[0])

        # 同理 range(scaled.shape[-1]) 沿着 last dimension 生成一个seq: layer的 neuron数 = #(filters)
# Note: 每一个filter(即CNN的neuron) 的output是: 对feature map 取mean！！！
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

# 和上面的差不多，除了dic存的是value
def update_coverage_value(input_data, model, model_layer_value):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # range(scaled.shape[-1])
        for no_neuron in range(scaled.shape[-1]):
            model_layer_value[(layer_names[i], no_neuron)] = np.mean(scaled[..., no_neuron])

    return intermediate_layer_outputs



def get_signature():
    now = datetime.now()
    past = datetime(2015, 6, 6, 0, 0, 0, 0)
    timespan = now - past
    time_sig = int(timespan.total_seconds() * 1000)

    return str(time_sig)
