# Demo

## Dependencies

Part of the needed dependencies are listed as below for the experiment
```
cuda 9.0
cudnn 7.1

Python 2.7.14
tensorflow-gpu (1.2.1)
Keras (2.2.4)
h5py (2.7.1)
Pillow (5.0.0)
opencv-python
matplotlib
```

To install on Linux(ubuntu)
```
When installing tensorflow 1.2.1, you can download the specified version on PYPI or the website of tensorflow.
pip install tensorflow-gpu
pip install keras
pip install matplotlib
pip install Pillow
pip install h5py
pip install opencv-python
```

## To run

generate adversarial examples for ImageNet
```
cd ImageNet
python gen_diff.py [2] 0.25 10 3 0.5 vgg16
#meanings of arguments
#python gen_diff.py 
[2] -> the list of neuron selection strategies
0.25 -> the activation threshold of a neuron
10 -> the number of neurons selected to cover
3 -> the number of times for mutation on each seed
0.5 -> lambda to balance 2 optimization objectives: higher, focus more on neuron coverage; lower, more adversarial
vgg16 -> the DL model under test
```

generate adversarial examples for MNIST
```
cd MNIST
python gen_diff.py [2] 0.5 5 3 0.5 model1
#meanings of arguments are the same as above
```