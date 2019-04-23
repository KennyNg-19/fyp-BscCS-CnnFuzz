
# coding: utf-8

# cmd with 7 params: python improved_method.py 2 0.2 30 3 5 4 [4123] 9
#  0.2 is good for LeNet-1               model_No(1/2/3)
# for LeNet 1, No.neurons to cover should be small, < 6 is normally ok

# basic setup & Import related modules
import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict
import matplotlib.pyplot as plt
import random, math
import time
import os
import sys
import cv2
from datetime import datetime

# import module
from utils import *
from IO_and_init import *
from mut_operators import *
# init all sample data sets, by randomly picking and do normalization：[0,255] -> [0,1]
(train_datas, train_labels), (test_datas, test_labels) = load_and_preprocess_data(10000, 2000) 

# init storage dir
seeds_dir = './seeds_improved_method/'
existing_imgs = [img for img in os.listdir(seeds_dir) if img.endswith(".png")]
run_more_mutation = False
if len(existing_imgs) > 0:
    more_mutation = input("keep the prev %d test cases. Not run operators? [y/other keys]: " % len(existing_imgs))
    if more_mutation != 'y' and more_mutation != 'Y':
        run_more_mutation = True
    print('---------read %d test cases from %s----------' %(len(existing_imgs), seeds_dir))
    pass
else:
    run_more_mutation = True # there is no test cases before



# load model
from keras.layers import Input
from keras.models import load_model

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
else:
    print("no model!!")

# model.summary()

if run_more_mutation:
    while True:
        mutation_ratios = []
        while True:
            operator_name = input("\nInput an operator_name among DR, LE, DM, DF, NP, AFRs, whi ,rot ,sh ,fl: ")
            if operator_name not in ["DR", "LE", "DM", "DF", "NP", "AFRs", 'whi' ,'rot' ,'sh' ,'fl']:
                print("Sorry, it is invalid...")
                continue
            else:
                while True:
                    while True:
                        try:
                            mutation_ratio = float(input('Input a ratio between 0 and 1:  '))
                            if mutation_ratio <= 1 and mutation_ratio >= 0:
                                mutation_ratios.append(mutation_ratio)
                                break
                            print("Fails, a ratio should be in [0,1]")
                            continue
                        except ValueError: # int() fails
                           print("That's not an number!")
                           continue

                    more_ratio = input("any more mutatio ratio[y/any other key]? ")
                    if more_ratio == 'y':
                        continue
                    break
                run_operator(mutation_ratios, operator_name, model, (train_datas, train_labels), test_datas, test_labels, seeds_dir)
                break
        more = input("any more operator[y/any other key]? ")
        if more == 'y':
            continue
        break



# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
print("\n------------------------Run improved testing method for MNIST-------------------------")

# set input dir: seeds
img_dir = './seeds_improved_method/'
img_names = [img for img in os.listdir(img_dir) if img.endswith(".png")] # return a list containing the NAMEs of only the img files in that dir path.
total_img_num = len(img_names)
random.shuffle(img_names)
while True:
    seeds_num = int(input("Input the number of filtered test cases <= %d:  " % total_img_num))
    if seeds_num <= 0 or seeds_num > total_img_num:
        print("Sorry, it has to be (0, %d] " % total_img_num)
        continue
    else:
        break
img_names = img_names[:seeds_num]

# set output dir: generated adversrials
save_dir = './gen_adversarial_improved_method/' + model_name + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" \
                + sys.argv[6] + "_" + sys.argv[7] + "_" + sys.argv[8] + "/"
init_storage_dir(save_dir)

# set metrics
# metric 1: basic neuron coverage 
model_layer_times1 = init_coverage_times(model)  # a dict for coverage times of each neuron covered 
model_layer_times2 = init_coverage_times(model)  # same as above, but update when new image and adversarial images found

model_layer_value1 = init_coverage_value(model) #

total_neuron_num = len(model_layer_times1) # constant

# metric 2: k-section coverage
multisection_num = int(sys.argv[6])
# a dict for each neuron outputs' ranges
model_neuron_values = load_file("%s_neuron_ranges.npy" % model_name) 
k_section_neurons_num = len(model_neuron_values)
k_multisection_coverage = init_multisection_coverage_value(model, multisection_num)
total_section_num = k_section_neurons_num * multisection_num # constant, of all neurons' sections


# metric 3: corner coverage
upper_corner_coverage = init_coverage_times(model)
lower_corner_coverage = init_coverage_times(model)

# set hyper params
threshold = float(sys.argv[2]) # activation threshold
target_neuron_cover_num = int(sys.argv[3]) # neurons to cover
iteration_times = int(sys.argv[4]) # 即 epoch
balance_lambda = float(sys.argv[5]) # Optimization λ, greater then focus on Neuron coverage; lese, on adversrial example
neuron_select_strategy = sys.argv[7] # among [1 2 3 4], pick one, or more
num_strategy = [x for x in neuron_select_strategy if x in ['1', '2', '3', '4']]
print("\nNeuron Selection Strategies: " + str(num_strategy),
    ", each aims at %d neurons" % int(target_neuron_cover_num / len(num_strategy)))

predict_weight = 0.5 # weight, in 1st part of optimization 
learning_step = 0.02

total_time = 0
total_norm = 0
total_adversrial_num = 0
total_perturb_adversrial = 0
total_adver_iterations = 0

wrong_predi = 0
find_adv_one_epoch = 0

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
print("\n------------------------------- Start Fuzzing(%d filtered seeds) --------------------------------" % seeds_num)
print("Store: generated adversarial saved in:", save_dir)
print("Note: to find adversarials with MINIMAL pertrubations, ONCE FOUND in %d epochs, the test will go to the next iteration\n" % iteration_times)
for i in range(seeds_num):

    start_time = time.process_time()
    #seed_list
    img_list = []


    img_name = os.path.join(img_dir,img_names[i]) # dir+name -> single img path, (name 即img_names[i])
    if (i + 1) % 10 == 0:
        print("Input "+ str(i+1) + "/" + str(seeds_num) + ": " + img_name)

    tmp_img = preprocess_image(img_name)
    img_list.append(tmp_img)
    orig_img = tmp_img.copy() # for later l2 distance comparison, diff_img = gen_img - orig_img


    # get right labels
    img_name = img_names[i].split('.')[0] # extract img name without the path suffix(after the “.”）
    right_label = int(img_name.split('_')[1]) # seed name is like "206_0", extract the label exactly from the 2nd part of the name

    # ----------------------------------------------------------------
    # coverage, for the original img, in model_layer_times2
    update_coverage(tmp_img, model, model_layer_times2, model_neuron_values, k_multisection_coverage, \
    multisection_num, upper_corner_coverage, lower_corner_coverage, threshold)

    # ====================================================================================================
    # ====================================================================================================
    # ====================================================================================================
    # fuzzing
    while len(img_list) > 0:
        # grab the head element
        gen_img = img_list[0] # (1, 28, 28, 1)
        img_list.remove(gen_img)

    #   for the 1st part of optimization function
        # get each class's scores, c and c_topk = dnn.predict(Xs)
        # first check if input already induces differences
        orig_pred = model.predict(gen_img)
        orig_pred_label = np.argmax(orig_pred[0])
        # label_top5 = np.argsort(orig_pred[0])[-5:]
        if orig_pred_label != right_label:
            wrong_predi += 1

        # neurons' values, coverage for gen_img, in model_layer_value1, as past testing
        update_coverage_value(gen_img, model, model_layer_value1)
        update_coverage(gen_img, model, model_layer_times1, model_neuron_values, k_multisection_coverage, \
            multisection_num, upper_corner_coverage, lower_corner_coverage, threshold)

        # prediction class score
        top_class_score = K.mean(model.get_layer('before_softmax').output[..., orig_pred_label])

        # other classes' scores
        other_classes_k = int(sys.argv[8])
        top_otherk_class_labels = np.argsort(orig_pred[0])[-other_classes_k:]
        top_otherk_class_scores = 0
        for i_class in range(2, other_classes_k + 1):
            top_otherk_class_scores += K.mean(model.get_layer('before_softmax').output[..., top_otherk_class_labels[-i_class]])


        # optimization obj function 1st part: sum(c_topk) - c， 
        # predefined: predict_weight = 0.5
        class_scores_loss = (predict_weight * top_otherk_class_scores - top_class_score)


    #  for the 2nd part of optimization function: λ · sum(neurons)
        # select neurons, selection(dnn, cov_tracker, strategies, m), based on past testing(model_layer_value/times1)
        neuron_ouput_loss = target_neurons_in_grad(model, model_layer_times1, model_layer_value1, \
                                       neuron_select_strategy, target_neuron_cover_num, threshold) 
        # neuron_ouput_loss = neuron_scale(neuron_ouput_loss) # useless, and negative result

    # the complete loss function
        # obj = sum(c_topk) - c + λ · sum(neurons)
        obj_function = class_scores_loss + balance_lambda * K.sum(neuron_ouput_loss) # neuron_ouput_loss is a list

    # for adversarial image generation
        final_loss = K.mean(obj_function)



    # set grads = @obj/@x
    # 1.define gradients backend function
        grads_tensor_list = []
        # grads_tensor_list = [class_score, loss]
        # grads_tensor_list.extend(neuron_ouput_loss) # extend a list

        # K.gradients(loss，vars)： compute derivaticve of loss wrt vars 
        # gradient obtained: compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, model.input)[0])
        grads_tensor_list.append(grads)

    # 2.compile function
        # K.function(inputs, outputs, updates=None, **kwargs): Instantiates a Keras function.
        # inputs: List of placeholder tensors        # outputs: List of output tensors.
        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], grads_tensor_list)

        # 3. run gradient ascent
        for iters in range(iteration_times): # iteration_times, a limit on the iteration time

            # run function, get gradient, in loss_list
            loss_list = iterate([gen_img])

            # perturbation = processing(grads)
            perturb = loss_list[-1] * learning_step
            # mutated input obtained
            gen_img += perturb


            # measure 1: improvement on coverage
            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]
            advers_pred = model.predict(gen_img) # scores
            advers_max_score = max(advers_pred[0]) # the score of the class with highest probability  
            advers_pred_label = np.argmax(advers_pred[0]) 
            
            # update cov_tracker
            update_coverage(gen_img, model, model_layer_times1, model_neuron_values, k_multisection_coverage, \
            multisection_num, upper_corner_coverage, lower_corner_coverage, threshold) # for seed selection

            current_coverage = neuron_covered(model_layer_times1)[2]

            # measure 2: l2 distance
            diff_img = gen_img - orig_img
            L2_norm = np.linalg.norm(diff_img)
            orig_L2_norm = np.linalg.norm(orig_img)
            perturb_adversrial = L2_norm / orig_L2_norm


            # Evaluate measures: if coverage improved by x′ is desired and l2_distance is small
            # print('coverage diff = %.3f, > %.4f? %s' % (current_coverage - previous_coverage, 0.01 / (i + 1), current_coverage - previous_coverage >  0.01 / (i + 1)))
            # print('perturb_adversrial = %f, < 0.01 %s' % (perturb_adversrial, perturb_adversrial < 0.1))
            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversrial < 0.02:
                print("======Find a good gen_img to imrpove NC and can be a new seed======")
                img_list.append(gen_img)

            # when find an adversrial, then break
            if advers_pred_label != orig_pred_label:
                if(iters == 0):
                    find_adv_one_epoch += 1

                total_adversrial_num += 1
                       

                update_coverage(gen_img, model, model_layer_times2, model_neuron_values, k_multisection_coverage, \
            multisection_num, upper_corner_coverage, lower_corner_coverage, threshold) # for seed selection

                total_norm += L2_norm

                total_perturb_adversrial += perturb_adversrial

                gen_img_tmp = gen_img.copy()

                gen_img_deprocessed = deprocess_image(gen_img_tmp)
                # write generated img to disk
                save_img_name = save_dir + str(total_adversrial_num) + model_name + "_" + \
                    str(orig_pred_label) + '_as_' + str(advers_pred_label) + "_" + "%.2f" % advers_max_score +'.png'

                cv2.imwrite(save_img_name, gen_img_deprocessed)

                
                # apply Grad-CAM algorithm to the generated adversrial examples
                heatmap = get_heatmap(model, advers_pred_label, gen_img, \
                    "block2_conv1", model.get_layer('block2_conv1').output_shape[-1])
                # write generated img with heatmap to disk
                impose_heatmap_to_img(save_img_name, save_dir, heatmap, total_adversrial_num, \
                    orig_pred_label, advers_pred_label)                
                
                # print("===========Find an adversrial, break============")
                break

        total_adver_iterations += iters

    if (i + 1) % 10 == 0:    
        print('NC: %d/%d <=> %.3f, ' % (len([v for v in model_layer_times2.values() if v > 0]),
        len(model_layer_times2), (neuron_covered(model_layer_times2)[2])), \
        "\ttotal adversarial generated: " + str(total_adversrial_num))
        print("incorrect predict %d/%d <=> %.2f" % (wrong_predi, seeds_num, wrong_predi/seeds_num))

    if (i + 1) % 60 == 0:
        covered_sections_num = 0
        for neuron_sections in k_multisection_coverage.values(): # each layer: {[0.0.0.0...], [0.0.0.0...], ...}
            for key in neuron_sections: # each neuron： neuron_sections [0.0.0.0...]
                if key > 0:
                    covered_sections_num += 1
        print('====================================')                            
        print('K-section coverage: %d/%d <=> %.3f' % (covered_sections_num, total_section_num, \
        covered_sections_num/total_section_num))

        print('UpperCorner coverage: %d/%d <=> %.3f' % (len([v for v in upper_corner_coverage.values() if v > 0]), \
        k_section_neurons_num, len([v for v in upper_corner_coverage.values() if v > 0])/k_section_neurons_num))
        # print('LowerCorner coverage: %d/%d <=> %.3f' % (len([v for v in lower_corner_coverage.values() if v > 0]), \
        # k_section_neurons_num, len([v for v in lower_corner_coverage.values() if v > 0])/k_section_neurons_num))
        print('====================================')            

    end_time = time.process_time()
    duration = end_time - start_time
    # print('Time : %.3f s\n' % duration)
    total_time += duration

print('\n--------------------------Summary-----------------------------')
print("wrong prediction(cross decision boundary) %d/%d <=> %.3f\n" % (wrong_predi, seeds_num, wrong_predi/seeds_num))
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
# print('LowerCorner coverage: %d/%d <=> %.3f' % (len([v for v in lower_corner_coverage.values() if v > 0]), \
# k_section_neurons_num, len([v for v in lower_corner_coverage.values() if v > 0])/k_section_neurons_num))
try:
    print('\ntotal adversrial num  = %d/%d chances(epochs)' % (total_adversrial_num, seeds_num))
    print('avg norm = %.3f ' % (total_norm / total_adversrial_num))
    # print('average time of generating an adversarial input %.3f s' % (total_time / total_adversrial_num))
    print('avg perb per adversrial = %.4f' % (total_perturb_adversrial / total_adversrial_num))
    print('total mutation iterations for these adversrials: %d (/%d)' % (total_adver_iterations, seeds_num * iteration_times))
    print('avg mutation iterations for a adversarial generation: %.2f' % (total_adversrial_num / total_adver_iterations))
except ZeroDivisionError:
    print('No adversrial is generated')
print('\ntotal time = %.3fs' % total_time)

# print("-----------K-section coverage-----------------")
# for layer_no, layer in enumerate(model.layers):
#     # 对于不经过activation的layer, 不考虑其coverage
#     if 'flatten' in layer.name or 'input' in layer.name:
#         continue

#     # 对于经过activation的layer
#     print("Layer %d: %s" % (layer_no + 1, layer.name))
#     for index in range(layer.output_shape[-1]): # 输出张量 last D
#         if (index + 1) % 4 != 1:
#             print("N %d: %s,   " % (index + 1, str(k_multisection_coverage[(layer.name, index)])), end = "")
#         else:
#             print("N %d: %s,   " % (index + 1, str(k_multisection_coverage[(layer.name, index)])))
#     print("\n")

# print("---------------------------------------------")
