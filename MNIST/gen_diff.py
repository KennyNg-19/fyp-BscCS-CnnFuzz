# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.layers import Input
from cv2 import imwrite
from utils_tmp import *
import sys
import os
import time

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3

def load_data(path="../MNIST_data/mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


# 选定模型和输入size
# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# define input tensor as a placeholder 实例化 Keras 张量
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)

model_name = sys.argv[6]

if model_name == 'model1':
    model1 = Model1(input_tensor=input_tensor)
    print("-------------------Testing on LeNet-1(52 neurons)---------------------")
elif model_name == 'model2':
    model1 = Model2(input_tensor=input_tensor)
    print("-------------------Testing on LeNet-4(148 neurons)---------------------")
elif model_name == 'model3':
    model1 = Model3(input_tensor=input_tensor)
    print("-------------------Testing on LeNet-5(268 neurons)---------------------")
else:
    print('please specify model name')
    os._exit(0)

# print(model1.name)

# model_layer_dict1 = init_coverage_tables(model1)
# 都是 deafult dict
model_layer_times1 = init_coverage_times(model1)  # times of each neuron covered 次数
model_layer_times2 = init_coverage_times(model1)  # 和上一行 初始化保持一致 直到 update when new image and adversarial images found

model_layer_value1 = init_coverage_value(model1) #






# 预设实验参数 包括 inputs
# img_names = image.list_pictures('../seeds_20', ext='JPEG')

img_dir = './seeds'
img_names = [img for img in os.listdir(img_dir) if img.endswith(".png")] # return a list containing the NAMEs of only the img files in that dir path.
img_num = len(img_names)

# e.g.[0,1,2] None for neurons not covered, 0 for covered often, 1 for covered rarely, 2 for high weights
neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
target_neuron_cover_num = int(sys.argv[3])
# subdir = sys.argv[4] # where to store the output
iteration_times = int(sys.argv[4]) # 即 epoch
neuron_to_cover_weight = float(sys.argv[5]) # Optimization 第二部分的λ 协调两部分: 越大，则以Neuron coverage为目标；越小，则以more adversial为目标

predict_weight = 0.5 # Optimization 第一部分的weight
learning_step = 0.02

# start = time.clock()
total_time = 0
total_norm = 0
adversial_num = 0

total_perturb_adversial = 0

# 设置 output 选定 storage dir
save_dir = './gen_adversarial/'

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

# if storage dir not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 开始实验
print("\n------------------------------- Start --------------------------------\n")
for i in range(img_num):

    start_time = time.process_time()
    #seed_list
    img_list = []


    img_name = os.path.join(img_dir,img_names[i]) # dir+name 合成single img path, (name 即img_names[i])
    print("Input "+ str(i+1) + "/" + str(img_num) + ": " + img_name)

    tmp_img = preprocess_image(img_name) # function, return a copy of the img in the path, 准备mutate -> gen_img
    img_list.append(tmp_img)

    orig_img = tmp_img.copy() # 比较mutation结果需要， diff_img = gen_img - orig_img


    # to get Label
    img_name = img_names[i].split('.')[0] # extract img name without the path suffix(after the “.”）
    mannual_label = int(img_name.split('_')[1]) # seed name is like "206_0", extract the label exactly from the 2nd part of the name

# ----------------------------------------------------------------
    # 原生img 输入，记下 原生nueron cover情况
    # model_layer_times2 ??
    update_coverage(tmp_img, model1, model_layer_times2, threshold)

    while len(img_list) > 0:
    	# grab the head element
        gen_img = img_list[0]
        img_list.remove(gen_img)




    #  Optimization 第一部分： 找到 c, c_topk = dnn.predict(Xs)
        # first check if input already induces differences
        pred1 = model1.predict(gen_img)
        label1 = np.argmax(pred1[0]) # [0] ??
        label_top5 = np.argsort(pred1[0])[-5:]

        # 记下 gen_img 对应的nueron value和cover 情况 ： 作为 past testing !!!
        update_coverage_value(gen_img, model1, model_layer_value1)
        update_coverage(gen_img, model1, model_layer_times1, threshold)

        orig_label = label1
        orig_pred = pred1

        # Tensor: (?,) first dimension is not fixed in the graph and it can vary between run calls
        loss_1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss_2 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-2]])
        loss_3 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-3]])
        loss_4 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-4]])
        loss_5 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-5]])
        # 第一部分，sum(c_topk) - c， hyper param: predict_weight = 0.5,
        layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)




    # Optimization 第二部分： neurons = selection(dnn, cov_tracker, strategies, m) 根据  past testing !!!
        # neuron coverage loss, in a List, 待cover neuron差的部分值！！！
        loss_neuron = neuron_selection(model1, model_layer_times1, model_layer_value1, # 代表  past testing
                                       neuron_select_strategy, target_neuron_cover_num, threshold) # 3 Hyper params
        # loss_neuron = neuron_scale(loss_neuron) # useless, and negative result


        # Optimization 目标函数
        # 第二部分, λ · sum(neurons)
        # extreme value means the activation value for a neuron can be as high as possible ... 增大第二部分的要求
        EXTREME_VALUE = False
        if EXTREME_VALUE:
            neuron_to_cover_weight = 2
        # else by default neuron_to_cover_weight = 0.5

        # obj = sum(c_topk) - c + λ · sum(neurons)
        layer_output += neuron_to_cover_weight * K.sum(loss_neuron) # loss_neuron is a list

        # for adversarial image generation
        final_loss = K.mean(layer_output) # 梯度的目标函数值




        # gradient obtained: compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
        grads_tensor_list.extend(loss_neuron) # extend 加一个list
        grads_tensor_list.append(grads)
        # this function returns the loss and grads given the input picture

        iterate = K.function([input_tensor], grads_tensor_list)



        the_input_adversarial_num = 0
        # we run gradient ascent for 3 steps
        for iters in range(iteration_times): # 1 epoch, 最多一个 adversrial generation ≤ iteration_times 输入超参epoch

            loss_neuron_list = iterate([gen_img])

            perturb = loss_neuron_list[-1] * learning_step

            # 生成 mutated x' 完毕
            gen_img += perturb


            # measure 1: improvement on coverage
            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]
            pred1 = model1.predict(gen_img)
            label1 = np.argmax(pred1[0])

            #  update cov_tracker
            update_coverage(gen_img, model1, model_layer_times1, threshold) # for seed selection
            current_coverage = neuron_covered(model_layer_times1)[2]

            # measure 2: l2 distance
            diff_img = gen_img - orig_img
            L2_norm = np.linalg.norm(diff_img)
            orig_L2_norm = np.linalg.norm(orig_img)
            perturb_adversial = L2_norm / orig_L2_norm


            # 检验效果: if coverage improved by x′ is desired and l2_distance is small
            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
                img_list.append(gen_img)
                # print('coverage diff = ', current_coverage - previous_coverage, 'perturb_adversial = ', perturb_adversial)

            if label1 != orig_label:
                update_coverage(gen_img, model1, model_layer_times2, threshold)

                total_norm += L2_norm

                total_perturb_adversial += perturb_adversial

                # print('L2 norm : ' + str(L2_norm))
                # print('ratio perturb = ', perturb_adversial)

                gen_img_tmp = gen_img.copy()

                gen_img_deprocessed = deprocess_image(gen_img_tmp)
                # use timestamp to name the generated adversial input
                save_img_name = save_dir + img_name + '_' + str(get_signature()) + '.png'

                imwrite(save_img_name, gen_img_deprocessed)

                the_input_adversarial_num += 1
                adversial_num += 1

    end_time = time.process_time()

    print('covered neurons percentage %.3f for %d neurons'
      % ((neuron_covered(model_layer_times2)[2]), len(model_layer_times2)))
    print('In %d epochs: %d adversarial examples' % (iteration_times, the_input_adversarial_num))

    duration = end_time - start_time

    print('Time : %.3f s\n' % duration)

    total_time += duration

print('\n--------------------------Summary-----------------------------')
print('covered neurons percentage %.3f for %d neurons'
      % ((neuron_covered(model_layer_times2)[2]), len(model_layer_times2)))
print('total time = ' + str(total_time) + 's')
print('total adversial num = ' + str(adversial_num))
print('average norm = %.3f ' % (total_norm / adversial_num))
print('average time of generating an adversarial input %.3f s' % (total_time / adversial_num))
print('average perb adversial = ' + str(total_perturb_adversial / adversial_num))
