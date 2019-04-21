from keras.datasets import cifar10

(train_datas, train_labels), (test_datas, test_labels) = cifar10.load_data()

train_datas = train_datas.astype('float32')
train_datas /= 255

print('train_datas shape:', train_datas.shape)


# init_neuron_values
from utils_tmp import *
from keras.models import load_model



modelName = "Model4"
model = load_model("%s.h5" % modelName)
model_neuron_values = init_neuron_values(model)
        
img_height, img_width = 32, 32
img_channels_num = 3

for i, single_training_example in enumerate(train_datas[:10000]):
    if i % 500 == 0:
        print(i)
    update_neuron_value(single_training_example.reshape(1, img_height, img_width, img_channels_num),\
                            model, model_neuron_values)


import numpy as np

# Save

np.save('%s_neuron_ranges.npy' % modelName, model_neuron_values) 
