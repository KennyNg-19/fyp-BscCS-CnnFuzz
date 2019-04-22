# -*- coding: utf-8 -*-

import random
import os
import cv2
import keras
from collections import defaultdict
from keras import backend as K
from keras.models import Model

from IO_and_init import *

model_layer_weights_top_k = []

# -------------------------------------------------------------------------------------------------------------------

# import mut_operators
# source_mut_opts = mut_operators.MutationOperators()

# ------------------------------------------------------------------------------------------------
def neuron_to_cover(not_covered,model_layer_dict):
    if not_covered: 
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
# access the dict of coverage times
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
    cv2.imwrite(save_heatmap_name, superimposed_img)
