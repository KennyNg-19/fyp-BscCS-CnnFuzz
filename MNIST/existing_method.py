# -*- coding: utf-8 -*-

# cmd with 7 params: python existing_method.py 2 0.25 8 3 5 4 [4123]
#                                         model_No(1/2/3)
# for LeNet 1, No.neurons to cover should be small, < 6 is normally ok

from __future__ import print_function

from keras.layers import Input
from cv2 import imwrite
from utils_tmp import *
import sys
import os
import time
from keras.models import load_model


# load multiple models sharing same input tensor
K.set_learning_phase(0)

model_name = sys.argv[1]


if model_name == '1':
    model_name = "Model1"
    model = load_model('./Model1.h5')
    print('LeNet-1(52 neurons) loaded')
elif model_name == '2':
    model_name = "Model2"
    model = load_model('./Model2.h5')
    print('LeNet-4(148 neurons) loaded')
elif model_name == '3':
    model_name = "Model3"
    model = load_model('./Model3.h5')
    print('LeNet-5(268 neurons) loaded')
        # break
else:
    print("No such a model!!")
    os._exit(0)

# model.summary()

print("------------------------Run existing testing method for MNIST set-------------------------")
# 都是 deafult dict
model_layer_times1 = init_coverage_times(model)  # dict for coverage times of each neuron covered 次数
model_layer_times2 = init_coverage_times(model)  # 和上一行 初始化保持一致 直到 update when new image and adversarial images found

model_layer_value1 = init_coverage_value(model) #

total_neuron_num = len(model_layer_times1) # constant

# a dict for each neuron' ranges
model_neuron_values = load_file("%s_neuron_ranges.npy" % model_name) 
k_section_neurons_num = len(model_neuron_values)


multisection_num = int(sys.argv[6])
k_multisection_coverage = init_multisection_coverage_value(model, multisection_num)
total_section_num = k_section_neurons_num * multisection_num # constant, of all neurons' sections

upper_corner_coverage = init_coverage_times(model)
lower_corner_coverage = init_coverage_times(model)


# 预设实验参数 包括 inputs
from random import shuffle

img_dir = './seeds_existing_method/'
img_names = [img for img in os.listdir(img_dir) if img.endswith(".png")] # return a list containing the NAMEs of only the img files in that dir path.
total_img_num = len(img_names)
shuffle(img_names)
# while True:
#     test_datas_num = int(input("Input a number of test datas <= %d:  " % total_img_num))
#     if test_datas_num > total_img_num:
#         print("Sorry, it has to be < %d" % total_img_num)
#         continue
#     else:
#         break
# img_names = img_names[:test_datas_num]
img_num = len(img_names)
test_img_num = img_num

# e.g.[0,1,2] None for neurons not covered, 0 for covered often, 1 for covered rarely, 2 for high weights
threshold = float(sys.argv[2])
target_neuron_cover_num = int(sys.argv[3])
# subdir = sys.argv[4] # where to store the output
iteration_times = int(sys.argv[4]) # 即 epoch
balance_lambda = float(sys.argv[5]) # Optimization 第二部分的λ 协调两部分: 越大，则以Neuron coverage为目标；越小，则以more adversrial为目标
neuron_select_strategy = sys.argv[7] # str，准备用for loop来split
print("\nNeuron Selection Strategies: " + str([x for x in neuron_select_strategy if x in ['1', '2', '3', '4']]))


predict_weight = 0.5 # Optimization 第一部分的weight
learning_step = 0.02

# start = time.clock()
total_time = 0
total_norm = 0
total_adversrial_num = 0
adversrial_num = 0

total_perturb_adversrial = 0

# 设置 output 选定 storage dir
save_dir = './gen_adversarial_existing_method/'

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

# if storage dir not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

seed_num = 0
wrong_predi = 0
find_adv_one_epoch = 0
print("\n------------------------------- Start Fuzzing(50 normal seeds) --------------------------------")
print("Store: generated adversarial saved in:", save_dir)
print("Note: to find adversarials with MINIMAL pertrubations, ONCE FOUND in %d epochs, the test will go to the next iteration\n" % iteration_times)
for i in range(img_num):

    start_time = time.process_time()
    #seed_list
    img_list = []


    img_name = os.path.join(img_dir,img_names[i]) # dir+name 合成single img path, (name 即img_names[i])
    if (i + 1) % 5 == 0:
        print("Input "+ str(i+1) + "/" + str(img_num) + ": " + img_name)

    tmp_img = preprocess_image(img_name) # function, return a copy of the img in the path, 准备mutate -> gen_img
    img_list.append(tmp_img)

    orig_img = tmp_img.copy() # 比较mutation结果需要， diff_img = gen_img - orig_img


    # to get labels
    img_name = img_names[i].split('.')[0] # extract img name without the path suffix(after the “.”）
    right_label = int(img_name.split('_')[1]) # seed name is like "206_0", extract the label exactly from the 2nd part of the name

    # ----------------------------------------------------------------
    # 原生img 输入，记下 原生nueron cover情况
    # model_layer_times2 ??
    update_coverage(tmp_img, model, model_layer_times2, model_neuron_values, k_multisection_coverage, \
            multisection_num, upper_corner_coverage, lower_corner_coverage, threshold) # for seed selection

    while len(img_list) > 0:
    	# grab the head element
        gen_img = img_list[0]
        img_list.remove(gen_img)




    #  Optimization 第一部分： 找到 c, c_topk = dnn.predict(Xs)
        # first check if input already induces differences
        orig_pred = model.predict(gen_img)
        orig_pred_label = np.argmax(orig_pred[0]) # [0] ??
        label_top5 = np.argsort(orig_pred[0])[-5:]

        # 记下 gen_img 对应的nueron value和cover 情况 ： 作为 past testing !!!
        update_coverage_value(gen_img, model, model_layer_value1)
        update_coverage(gen_img, model, model_layer_times1, model_neuron_values, k_multisection_coverage, \
            multisection_num, upper_corner_coverage, lower_corner_coverage, threshold) # for seed selection

        seed_num += 1
        if orig_pred_label != right_label:
            wrong_predi += 1
            # print("----------------For a seed img %d: %d, model predicts %d, wrong------------" % (i+1, right_label, orig_pred_label))

        top_k_class = int(sys.argv[8])
        label_topk = np.argsort(orig_pred[0])[-top_k_class:]

        loss_1 = K.mean(model.get_layer('before_softmax').output[..., orig_pred_label])
        loss = 0
        # Tensor: (?,) first dimension is not fixed in the graph and it can vary between run calls
        for i_class in range(2, top_k_class + 1):
            loss += K.mean(model.get_layer('before_softmax').output[..., label_topk[-i_class]])


        # Optimization 第一部分，sum(c_topk) - c， hyper param: predict_weight = 0.5,
        layer_output = (predict_weight * loss - loss_1)



        # Optimization 第二部分： neurons = selection(dnn, cov_tracker, strategies, m) 根据  past testing !!!
        #  λ · sum(neurons)
        # extreme value means the activation value for a neuron can be as high as possible ... 增大第二部分的要求
        # neuron coverage loss, in a List, 待cover neuron差的部分值！！！
        loss_neuron = target_neurons_in_grad(model, model_layer_times1, model_layer_value1, # 代表  past testing
                                       neuron_select_strategy, target_neuron_cover_num, threshold) # the 3 Hyper params
        # loss_neuron = neuron_scale(loss_neuron) # useless, and negative result


    # 完整的 Optimization 目标函数
        # obj = sum(c_topk) - c + λ · sum(neurons)
        layer_output += balance_lambda * K.sum(loss_neuron) # loss_neuron is a list

    # for adversarial image generation
        final_loss = K.mean(layer_output) # 梯度的目标函数值



    # ---------------------------------------------------grads = @obj/@xs--------------------------------------------------------------
    # 1.定义 gradients backend函数: 求损失函数关于变量的导数，也就是网络的反向计算过程。
        grads_tensor_list = []
        # grads_tensor_list = [loss_1, loss]
        # grads_tensor_list.extend(loss_neuron) # extend 加一个list

        # K.gradients（loss，vars）： 用于求loss关于vars 的导数（梯度）(若为vars tensor，则是求每个var的偏导数,输出也是gradients tensor)----通过tensorflow的tf.gradients()
        # gradient obtained: compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, model.input)[0])
        grads_tensor_list.append(grads)

    # 2.编译 gradient函数：将一个计算图（计算关系）编译为具体的函数。典型的使用场景是输出网络的中间层结果
        # K.function(inputs, outputs, updates=None, **kwargs): Instantiates a Keras function.
        # inputs: List of placeholder tensors.
        # outputs: List of output tensors.
        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], grads_tensor_list) # 这里因为grads_tensor_list 有 grads，所激素和i在编译K.gradients这个函数



        the_input_adversarial_num = 0
        # we run gradient ascent for 3 steps
        for iters in range(iteration_times): # 1 epoch, 最多一个 adversrial generation ≤ iteration_times 输入超参epoch

            # run gradient函数, gradient, in grads_tensor_list, obtained
            loss_neuron_list = iterate([gen_img])

            # perturbation = processing(grads)
            perturb = loss_neuron_list[-1] * learning_step
            # mutated input obtained
            gen_img += perturb


            # measure 1: improvement on coverage
            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]
            advers_pred = model.predict(gen_img) # score in [0:1]
            advers_pred_label = np.argmax(advers_pred[0]) 
            


            #  update cov_tracker
            update_coverage(gen_img, model, model_layer_times1, model_neuron_values, k_multisection_coverage, \
            multisection_num, upper_corner_coverage, lower_corner_coverage, threshold) # for seed selection

            current_coverage = neuron_covered(model_layer_times1)[2]

            # measure 2: l2 distance
            diff_img = gen_img - orig_img
            L2_norm = np.linalg.norm(diff_img)
            orig_L2_norm = np.linalg.norm(orig_img)
            perturb_adversrial = L2_norm / orig_L2_norm


            # 检验效果: if coverage improved by x′ is desired and l2_distance is small
            # print('coverage diff = %.3f, > %.4f? %s' % (current_coverage - previous_coverage, 0.01 / (i + 1), current_coverage - previous_coverage >  0.01 / (i + 1)))
            # print('perturb_adversrial = %f, < 0.01 %s' % (perturb_adversrial, perturb_adversrial < 0.1))
            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversrial < 0.02:
                print("======Find a good gen_img to imrpove NC and can be a new seed======")
                img_list.append(gen_img)

            # Find an adversrial, break 否？
            if advers_pred_label != orig_pred_label:
                if(iters == 0):
                    find_adv_one_epoch += 1

                total_adversrial_num += 1
                the_input_adversarial_num += 1
                adversrial_num += 1

                update_coverage(gen_img, model, model_layer_times2, model_neuron_values, k_multisection_coverage, \
            multisection_num, upper_corner_coverage, lower_corner_coverage, threshold) # for seed selection

                total_norm += L2_norm

                total_perturb_adversrial += perturb_adversrial

                # print('L2 norm : ' + str(L2_norm))
                # print('ratio perturb = ', perturb_adversrial)

                gen_img_tmp = gen_img.copy()

                gen_img_deprocessed = deprocess_image(gen_img_tmp)
                # use timestamp to name the generated adversrial input
                save_img_name = save_dir + str(total_adversrial_num) + "_" + \
                    str(orig_pred_label) + '_as_' + str(advers_pred_label) + '.png'

                imwrite(save_img_name, gen_img_deprocessed)

                
                # apply Grad-CAM algorithm to the generated adversrial examples
                heatmap = apply_Grad_CAM(model, advers_pred_label, gen_img, \
                    "block2_conv1", model.get_layer('block2_conv1').output_shape[-1])
                import cv2
                # We resize the heatmap to have the same size as the original image
                img = cv2.imread(save_img_name)                
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                # We convert the heatmap to RGB
                heatmap = np.uint8(255 * heatmap)
                # We apply the heatmap to the original image            
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)                
                # 0.4 here is a heatmap intensity factor
                superimposed_img = heatmap * 0.4 + img
                # Save the image to disk
                save_heatmap_name = save_dir + str(total_adversrial_num) + "_" + \
                    str(orig_pred_label) + '_as_' + str(advers_pred_label) + '_heat.png'
                imwrite(save_heatmap_name, superimposed_img)
                

                # break ?
                # print("===========Find an adversrial, break============")
                break

    if (i + 1) % 5 == 0:    
        print('NC: %d/%d <=> %.3f' % (len([v for v in model_layer_times2.values() if v > 0]),
        len(model_layer_times2), (neuron_covered(model_layer_times2)[2])))
        # print("No. adversarial: " + str(adversrial_num))
        # # print('In %d epochs: %d adversarial examples' % (iteration_times, the_input_adversarial_num))
        # print("wrong predict %d/%d <=> %.3f" % (wrong_predi, seed_num, wrong_predi/seed_num))
    
    end_time = time.process_time()
    duration = end_time - start_time
    # print('Time : %.3f s\n' % duration)
    total_time += duration

print('\n--------------------------Summary-----------------------------')
print("wrong prediction(normal input data, should be small) %d/%d <=> %.3f\n" % (wrong_predi, seed_num, wrong_predi/seed_num))
# print('adversarial found in 1st epoch(close to decision boundary): %d/%d <=> %.2f\n' % (find_adv_one_epoch, test_img_num, find_adv_one_epoch/test_img_num))

print('covered neurons percentage %.3f for %d neurons'
      % ((neuron_covered(model_layer_times2)[2]), total_neuron_num))

covered_sections_num = 0
for neuron_sections in k_multisection_coverage.values(): # each layer: {[0.0.0.0...], [0.0.0.0...], ...}
    for key in neuron_sections: # each neuron： neuron_sections [0.0.0.0...]
        if key > 0:
            covered_sections_num += 1
print('%d-section coverage: %d/%d <=> %.3f' % (multisection_num, covered_sections_num, total_section_num, \
covered_sections_num/total_section_num))
print('UpperCorner coverage: %d/%d <=> %.3f' % (len([v for v in upper_corner_coverage.values() if v > 0]), \
k_section_neurons_num, len([v for v in upper_corner_coverage.values() if v > 0])/k_section_neurons_num))
print('LowerCorner coverage: %d/%d <=> %.3f' % (len([v for v in lower_corner_coverage.values() if v > 0]), \
k_section_neurons_num, len([v for v in lower_corner_coverage.values() if v > 0])/k_section_neurons_num))
try:
    print('\ntotal adversrial num  = %d/%d chances(epochs)' % (total_adversrial_num, test_img_num))
    print('average norm = %.3f ' % (total_norm / total_adversrial_num))
    # print('average time of generating an adversarial input %.3f s' % (total_time / total_adversrial_num))
    print('average perb adversrial = %.4f' % (total_perturb_adversrial / total_adversrial_num))
except ZeroDivisionError:
    print('No adversrial is generated')
print('\ntotal time = %.3fs' % total_time)
