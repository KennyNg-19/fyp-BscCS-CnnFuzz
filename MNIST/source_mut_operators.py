import tensorflow as tf
import numpy as np
import keras

import random
import math
import utils

class SourceMutationOperatorsUtils():

    def __init__(self):
        self.LR_mut_candidates = ['Dense', 'BatchNormalization']
        self.LA_mut_candidates = [keras.layers.ReLU(), keras.layers.BatchNormalization()]
        self.utils = utils.GeneralUtils()

    def LA_get_random_layer(self):
        num_of_LA_candidates = len(self.LA_mut_candidates) # [keras.layers.ReLU(), keras.layers.BatchNormalization()]
        random_index = random.randint(0, num_of_LA_candidates - 1)
        return self.LA_mut_candidates[random_index]

    def LR_model_scan(self, model):
        index_of_suitable_layers = []
        layers = [l for l in model.layers]

        for index, layer in enumerate(layers):
            layer_name = type(layer).__name__

            # condition 1: mainly focuses on layers ['Dense', 'BatchNormalization']
            # whose deletion does not make too much difference on the mutated model.
            is_in_candidates = layer_name in self.LR_mut_candidates

            # condition 2: only select the layer with the same input and output
            has_same_input_output_shape = layer.input.shape.as_list() == layer.output.shape.as_list()

            should_be_removed = is_in_candidates and has_same_input_output_shape

            if index == 0 or index == (len(model.layers) - 1):
                continue

            if should_be_removed:
                index_of_suitable_layers.append(index)
        return index_of_suitable_layers

    def LAs_model_scan(self, model):
        index_of_suitable_spots = []
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            if index == (len(model.layers) - 1):
                continue

            index_of_suitable_spots.append(index)
        return index_of_suitable_spots

    def AFRs_model_scan(self, model):
        index_of_suitable_spots = []
        layers = [l for l in model.layers]
        for index, layer in enumerate(layers):
            if index == (len(model.layers) - 1):
                continue
            try:
                if layer.activation is not None:
                    index_of_suitable_spots.append(index)
            except:
                pass
        return index_of_suitable_spots

class SourceMutationOperators():

    def __init__(self):
        self.utils = utils.GeneralUtils()
        self.check = utils.ExaminationalUtils()
        self.model_utils = utils.ModelUtils()
        self.SMO_utils = SourceMutationOperatorsUtils()

# mutation_ratio 只是决定变化的量, 在这一行 number_of_XXX_datas = math.floor(number_of_train_data * mutation_ratio)
#  np.random.permutation 打乱顺序

    #  Data repetition: mutate_ratio对acc 影响很小
    def DR_mut(self, train_dataset, model, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'DR')
        train_datas, train_labels = train_dataset
        self.check.mutation_ratio_range_check(mutation_ratio)
        self.check.training_dataset_consistent_length_check(train_datas, train_labels)

        # shuffle the original train data
        shuffled_train_datas, shuffled_train_labels = self.utils.shuffle_in_uni(train_datas, train_labels)

        # select a portion of data and reproduce
        number_of_train_data = len(train_datas)
        number_of_duplicate = math.floor(number_of_train_data * mutation_ratio)
        print("ramdomly repeat %d training datasets, totally %d" % (number_of_duplicate, number_of_train_data + number_of_duplicate))

        repeated_train_datas = shuffled_train_datas[:number_of_duplicate]
        repeated_train_labels = shuffled_train_labels[:number_of_duplicate]
        repeated_train_datas = np.append(train_datas, repeated_train_datas, axis=0)
        repeated_train_labels = np.append(train_labels, repeated_train_labels, axis=0)
        return (repeated_train_datas, repeated_train_labels), deep_copied_model

    # 2. LE (Label Error) 负相关 持续性影响很大: mutate_ratio越高，acc越低， acc可能小于0.5
    def LE_mut(self, train_dataset, model, label_lower_bound, label_upper_bound, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'LE')
        train_datas, train_labels = train_dataset
        LE_train_datas, LE_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)
        self.check.training_dataset_consistent_length_check(LE_train_datas, LE_train_labels)

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

    # 3. DM data missing: 轻度负相关, 但是mutate_ratio高于0.8后，acc下降更快
    def DM_mut(self, train_dataset, model, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'DM')
        train_datas, train_labels = train_dataset
        DM_train_datas, DM_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)
        self.check.training_dataset_consistent_length_check(DM_train_datas, DM_train_labels)

        number_of_train_data = len(DM_train_datas)
        number_of_deletion = math.floor(number_of_train_data * mutation_ratio)
        print("ramdomly delete %d training datasets: now %d datasets" % (number_of_deletion, number_of_train_data - number_of_deletion))

        shuffled_indexes = np.random.permutation(number_of_train_data)
        shuffled_indexes = shuffled_indexes[:number_of_deletion]


        # delete the selected data
        DM_train_datas = np.delete(DM_train_datas, shuffled_indexes, 0)
        DM_train_labels = np.delete(DM_train_labels, shuffled_indexes, 0)
        return (DM_train_datas, DM_train_labels), deep_copied_model

    # 4. DF (Data Shuffle) 影响太小 和 data repetition差不多
    def DF_mut(self, train_dataset, model, mutation_ratio):
        deep_copied_model = self.model_utils.model_copy(model, 'DF')
        train_datas, train_labels = train_dataset
        DF_train_datas, DF_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)
        self.check.training_dataset_consistent_length_check(DF_train_datas, DF_train_labels)

        number_of_train_data = len(DF_train_datas)
        number_of_shuffled_datas = math.floor(number_of_train_data * mutation_ratio)
        print("shuffle %d training datasets/%d datasets totally" % (number_of_shuffled_datas, number_of_train_data))

        shuffled_indexes = np.random.permutation(number_of_train_data)
        shuffled_indexes = shuffled_indexes[:number_of_shuffled_datas]

        DF_train_datas, DF_train_labels = self.utils.shuffle_in_uni_with_permutation(DF_train_datas, DF_train_labels, shuffled_indexes)
        return (DF_train_datas, DF_train_labels), deep_copied_model

    # 5.NP Noise Perturb, 负相关: mutate_ratio越高，acc越低; 同时,STD标准差越大，random_noise越大, acc越低
    def NP_mut(self, train_dataset, model, mutation_ratio, STD=0.1):
        deep_copied_model = self.model_utils.model_copy(model, 'NP')
        train_datas, train_labels = train_dataset
        NP_train_datas, NP_train_labels = train_datas.copy(), train_labels.copy()
        self.check.mutation_ratio_range_check(mutation_ratio)
        self.check.training_dataset_consistent_length_check(NP_train_datas, NP_train_labels)

        number_of_train_data = len(NP_train_datas)
        number_of_noise_perturbs = math.floor(number_of_train_data * mutation_ratio) # a specific amount of samples is chosen
        print("shuffle %d training datasets/%d datasets totally with STD %f" % (number_of_noise_perturbs, number_of_train_data, STD))

        shuffled_indexes = np.random.permutation(number_of_train_data) # a list of INDEX!! Parameters: int or array_like If x is an integer, randomly permute np.arange(x).
        shuffled_indexes_to_add_noise = shuffled_indexes[:number_of_noise_perturbs] # a list of index!!! 取打乱顺序后的，0-ratio决定的部分

        # 每个img in train_data都有一个 对应的随机生成的perturbation
        random_noise = np.random.standard_normal(NP_train_datas.shape) * STD # Draw samples from a standard Normal distribution (mean=0, stdev=1)
        for new_index in shuffled_indexes_to_add_noise: # new_index 是chosen sample的index
            # 所以NP_train_datas[new_index]就是对应chosen的image，random_noise[new_index]是他对应的随机生成的perturbation
            NP_train_datas[new_index] += random_noise[new_index]
        return (NP_train_datas, NP_train_labels), deep_copied_model




# --------------------------------------------------------------------------------------------------------------------------
    # # 6. Layer Removel: mimics the case that a line of code representing a DNN layer is removed by the developer.
    # def LR_mut(self, train_dataset, model, mutated_layer_indices=None):
    #     # Copying and some assertions
    #     deep_copied_model = self.model_utils.model_copy(model, 'LR')
    #     train_datas, train_labels = train_dataset
    #     copied_train_datas, copied_train_labels = train_datas.copy(), train_labels.copy()
    #     self.check.training_dataset_consistent_length_check(copied_train_datas, copied_train_labels)
    #
    #     # Randomly select from suitable layers instead of the first one
    #     index_of_suitable_layers = self.SMO_utils.LR_model_scan(model)
    #     number_of_suitable_layers = len(index_of_suitable_layers)
    #     if number_of_suitable_layers == 0:
    #         print('No layers be removed bcz LR only removes the layer with the same input and output, but there is no such one')
    #         return (copied_train_datas, copied_train_labels), deep_copied_model
    #
    #     new_model = keras.models.Sequential()
    #     layers = [l for l in deep_copied_model.layers]
    #
    #     if mutated_layer_indices == None:
    #         # 且一般 index_of_suitable_layers是 [], 所以基本都是无效的， new_model就是原model
    #         random_picked_layer_index = index_of_suitable_layers[random.randint(0, number_of_suitable_layers-1)]
    #         print('Selected layer by LR mutation operator', random_picked_layer_index)
    #
    #         for index, layer in enumerate(layers):
    #             if index == random_picked_layer_index:
    #                 continue
    #             new_model.remove(layer)
    #     else:
    #         # remove 指定的layer, 影响可能很大
    #         self.check.in_valid_indices_check(index_of_suitable_layers, mutated_layer_indices)
    #
    #         for index, layer in enumerate(layers):
    #             if index in mutated_layer_indices:
    #                 continue
    #             new_model.remove(layer)
    #
    #     return (copied_train_datas, copied_train_labels), new_model
    #
    #
    # # 7. focuses on adding layers like Activation, BatchNormalization,
    # # which introduces possible faults caused by adding or duplicating a line of code representing a DNN layer.
    # def LAs_mut(self, train_dataset, model, mutated_layer_indices=None):
    #     # Copying and some assertions
    #     deep_copied_model = self.model_utils.model_copy(model, 'LAs')
    #     train_datas, train_labels = train_dataset
    #     copied_train_datas, copied_train_labels = train_datas.copy(), train_labels.copy()
    #     self.check.training_dataset_consistent_length_check(copied_train_datas, copied_train_labels)
    #
    #     # Randomly select from suitable spots instead of the first one
    #     index_of_suitable_spots = self.SMO_utils.LAs_model_scan(model)
    #     number_of_suitable_spots = len(index_of_suitable_spots)
    #     if number_of_suitable_spots == 0:
    #         print('No layers be added')
    #         print('There is no suitable spot for the input model')
    #         return (copied_train_datas, copied_train_labels), deep_copied_model
    #
    #     new_model = keras.models.Sequential()
    #     layers = [l for l in deep_copied_model.layers]
    #
    #     if mutated_layer_indices == None: # 手动选定layer index, 待下面会check是否合适
    #         random_picked_spot_index = index_of_suitable_spots[random.randint(0, number_of_suitable_spots-1)]
    #         print('Selected layer by LAs mutation operator', random_picked_spot_index)
    #
    #         for index, layer in enumerate(layers):
    #
    #             if index == random_picked_spot_index:
    #                 new_model.add(layer)
    #                 new_model.add(self.SMO_utils.LA_get_random_layer())
    #                 continue
    #             new_model.add(layer)
    #     else:
    #         self.check.in_valid_indices_check(index_of_suitable_spots, mutated_layer_indices)
    #
    #         for index, layer in enumerate(layers):
    #             if index in mutated_layer_indices:
    #                 new_model.add(layer)
    #                 new_model.add(self.SMO_utils.LA_get_random_layer())
    #                 continue
    #             new_model.add(layer)
    #
    #
    #     return (copied_train_datas, copied_train_labels), new_model

    # The AFRs operator randomly removes all the activation functions of a layer,
    # to mimic the situation that the developer forgets to add the activation layers.
    def AFRs_mut(self, train_dataset, model, mutated_layer_indices=None):
        # Copying and some assertions
        deep_copied_model = self.model_utils.model_copy(model, 'AFRs')
        train_datas, train_labels = train_dataset
        copied_train_datas, copied_train_labels = train_datas.copy(), train_labels.copy()
        self.check.training_dataset_consistent_length_check(copied_train_datas, copied_train_labels)

        # Randomly select from suitable layers instead of the first one
        index_of_suitable_layers = self.SMO_utils.AFRs_model_scan(model)
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
            self.check.in_valid_indices_check(index_of_suitable_layers, mutated_layer_indices)
            for index, layer in enumerate(layers):
                if index in mutated_layer_indices:
                    layer.activation = lambda x: x  # change into a linear function f(x) = x
                    new_model.add(layer)
                    continue
                new_model.add(layer)

        return (copied_train_datas, copied_train_labels), new_model
