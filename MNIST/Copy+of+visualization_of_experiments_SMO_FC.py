
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


# import os

# # 此处为google drive中的文件路径,drive为之前指定的工作根目录，要加上
# os.chdir("/content/drive/My Drive/DeepMutationOperators-master") 


# In[3]:


# Basic setup & Import related modules 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import keras 
import random, math

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# storage
import os
from cv2 import imwrite
from datetime import datetime

save_dir = './seeds_50/'

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

# if storage dir not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
def deprocess_image(x):
    # de-normalization: [0,1] -> [0,255]     
    x *= 255
    return x.reshape(28,28) #  for one test_case,original shape (784,)
  
counter_of_each_test_case = 1


# In[4]:


import utils, network, source_mut_operators 
utils = utils.GeneralUtils()
network = network.FCNetwork()
source_mut_opts = source_mut_operators.SourceMutationOperators()
 
(train_datas, train_labels), (test_datas, test_labels) = network.load_data() # load_data里 normalization：[0,255] -> [0,1] 

print('train_datas shape:', train_datas.shape)
print('test_datas shape:', test_datas.shape)
print('train_labels shape:', train_labels.shape)
print('test_labels shape:', test_labels.shape)

mutation_ratios = [i*0.05 + 0.05 for i in range(5,10)]

mutant_accuracy_control_threshold = [0.92 for i in mutation_ratios if i < 0.5] + [0.88 for i in mutation_ratios if i >= 0.5]

# initialize as an empty numpy array 
gen_test_cases = np.zeros((1, 784))
gen_test_cases = gen_test_cases[:-1, :] # remove the only line


# In[ ]:


# 1. DR (Data Repetition)
print("\n1. DR (Data Repetition) mutation operator")
difference_scores = []
counter = 0
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.DR_mut((train_datas, train_labels), model, mutation_ratio)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)

    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)
   
    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      counter += 1
      difference_scores.append(0)
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
#     predictions = trained_model.predict(test_datas) # right labels
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sDataRepetition%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))
          
print("now there are %d test cases" % gen_test_cases.shape[0])   


# In[ ]:


plt.title('DR mutation deference')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.1])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 2. LE (Label Error)
print("\n2. LE (Label Error) mutation operator")
difference_scores = []
counter = 0
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.LE_mut((train_datas, train_labels), model, 0, 9, mutation_ratio)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)
    
    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      counter += 1
      difference_scores.append(0)
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sLabelError%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))

    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0]) 


# In[ ]:


plt.title('LE mutation accurancy')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 3. DM (Data Missing)
print("\n3. DM (Data Missing) mutation operator")
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.DM_mut((train_datas, train_labels), model, mutation_ratio)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)
    
    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      counter += 1
      difference_scores.append(0)
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sDataMissing%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])   
    


# In[ ]:


plt.title('DM mutation difference')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 4. DF (Data Shuffle)
print("\n4. DF (Data Shuffle) mutation operator")
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.DF_mut((train_datas, train_labels), model, mutation_ratio)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)
    
    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      counter += 1
      difference_scores.append(0)
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sDataShuffle%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])  
    


# In[ ]:


plt.title('DF mutation difference')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 5. NP - Noise Perturb, STD=0.1
print("\n5. NP - Noise Perturb, STD=0.1 mutation operator")
STD = 0.1
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.NP_mut((train_datas, train_labels), model, mutation_ratio, STD=STD)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)
  
    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      difference_scores.append(0)
      counter += 1
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sNoisePerturSTD0.1%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])  
    


# In[ ]:


plt.title('NP with STD = 0.1 mutation difference')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 5. NP - Noise Perturb, STD = 1
print("\n5. NP - Noise Perturb, STD=1 mutation operator")
STD = 1
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.NP_mut((train_datas, train_labels), model, mutation_ratio, STD=STD)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)
    
    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      difference_scores.append(0)
      counter += 1
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f" % (counter, mutant_acc))
    

    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sNoisePerturSTD1%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])  
    


# In[ ]:


plt.title('NP with STD = 1 mutation difference')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 5. NP - Noise Perturb, STD=10
print("\n5. NP - Noise Perturb, STD=10 mutation operator")
STD = 10
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.NP_mut((train_datas, train_labels), model, mutation_ratio, STD=STD)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)

    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      difference_scores.append(0)
      counter += 1
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sNoisePerturSTD10%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])  
    


# In[ ]:


plt.title('NP with STD = 10 mutation difference')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# Syntax: copied_dataset(实际不变), mutated_model  = source_mut_opts.XX_mut(training_dataset, model)
# 6.LR - Layer Removal
print("\n6.LR - Layer Removal mutation operator")
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.LR_mut((train_datas, train_labels), model)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)

    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      difference_scores.append(0)
      counter += 1
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sLayerRemoval%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])  


# In[ ]:


plt.title('LR - Layer Removal')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 7. LAs - Layer Addition
print("\n7. LAs - Layer Addition mutation operator")
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.LAs_mut((train_datas, train_labels), model)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)

    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      difference_scores.append(0)
      counter += 1
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sLayerAddition%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])  


# In[ ]:


plt.title('LA - Layer Addition')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()


# In[ ]:


# 8. AFRs - Activation Function Removal
print("\n8. AFRs - Activation Function Removal mutation operator")
counter = 0
difference_scores = []
for mutation_ratio in mutation_ratios:
    model = network.create_normal_FC_model()
    (mutated_datas, mutated_results), mutated_model = source_mut_opts.AFRs_mut((train_datas, train_labels), model)

    model = network.compile_model(model)
    mutated_model = network.compile_model(mutated_model)
    
    trained_model = network.train_model(model, train_datas, train_labels)
    trained_mutated_model = network.train_model(mutated_model, mutated_datas, mutated_results)

    # quality control of mutant model
    mutant_loss, mutant_acc = trained_mutated_model.evaluate(test_datas, test_labels, verbose=False)
    if mutant_acc < mutant_accuracy_control_threshold[counter]:
      difference_scores.append(0)
      counter += 1
      print("the %dth DL mutant behaves badly, with acc %.3f and %.2f mutation ratio, will be dropped out\n" % (counter, mutant_acc, mutation_ratio))
      continue
    print("the %dth DL mutant passes the quality test, with acc %.3f and %.2f mutation ratio" % (counter + 1, mutant_acc, mutation_ratio))
    
    # get prediction results
    # predictions = trained_model.predict(test_datas)
    mutant_predictions = trained_mutated_model.predict(test_datas)
    
    right_labels = np.argmax(test_labels, axis=1) # right answer
    mutant_predi_labels = np.argmax(mutant_predictions, axis = 1)
    
    difference_indexes = np.nonzero(right_labels-mutant_predi_labels)[0] #  np.nonzero(right_labels-mutant_predi_labels) is a tuple with only one element: the [0],a numpy array
    difference_score = len(difference_indexes) / right_labels.size 
    difference_scores.append(difference_score) 
    
    counter += 1
    print("%d iter: the difference ratio: %.3f  with mutation ratio: %.2f" % (counter, difference_score, mutation_ratio))
    prev_gen_test_cases = gen_test_cases
    additional_test_cases = test_datas[difference_indexes]
    gen_test_cases = np.unique(np.append(gen_test_cases, additional_test_cases,axis = 0), axis=0) # difference_indexes is indices of test cases with differences
    
    # store test cases in to local folder
    for index, test_case in enumerate(additional_test_cases):
        imwrite('%sActivationFunctionRemoval%d_%d.png' % (save_dir, index + 1, right_labels[difference_indexes[index]]), deprocess_image(test_case))
        
    print(" test cases + %d\n" % (gen_test_cases.shape[0] - prev_gen_test_cases.shape[0]))

print("now there are %d test cases" % gen_test_cases.shape[0])  


# In[ ]:


plt.title('Activation Function Removal')
plt.axis([mutation_ratios[0], mutation_ratios[-1], 0, 0.2])
plt.plot(mutation_ratios, difference_scores)
plt.xlabel('Mutation ratio')
plt.ylabel('Differences')
plt.show()

