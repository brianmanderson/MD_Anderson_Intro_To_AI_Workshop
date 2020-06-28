#%% md

# Liver Model

#%% md

## Import some things

#%%

def return_sys_path():
    path = '.'
    for _ in range(5):
        if 'Deep_Learning' in os.listdir(path):
            break
        else:
            path = os.path.join(path,'..')
    return path
def return_data_path():
    path = '.'
    for _ in range(5):
        if 'Data' in os.listdir(path):
            break
        else:
            path = os.path.join(path,'..')
    return path

#%%

import os, sys
sys.path.append(return_sys_path())
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import *
from Deep_Learning.Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_Image_Scroll_Bar_Image
import tensorflow as tf
import tensorflow.keras.backend as K


base = return_sys_path()
data_path = os.path.join(return_data_path(),'Data','Niftii_Arrays','Records')
train_path = [os.path.join(data_path,'Train')]
validation_path = [os.path.join(data_path,'Validation')]
test_path = os.path.join(data_path,'Test')
model_path = os.path.join(base,'Models')
if not os.path.exists(model_path):
    os.makedirs(model_path)


image_size = 128
wanted_keys={'inputs':['image'],'outputs':['annotation']}
image_processors_train = [Expand_Dimensions(axis=-1),
                          Ensure_Image_Proportions(image_size,image_size),
                          Repeat_Channel(repeats=3),
                          Normalize_Images(mean_val=78,std_val=29),
                          Threshold_Images(lower_bound=-3.55,upper_bound=3.55),
                          Return_Outputs(wanted_keys)]
image_processors_validation = [Expand_Dimensions(axis=-1),
                          Ensure_Image_Proportions(image_size,image_size),
                          Repeat_Channel(repeats=3),
                          Normalize_Images(mean_val=78,std_val=29),
                          Threshold_Images(lower_bound=-3.55,upper_bound=3.55),
                          Return_Outputs(wanted_keys)]

batch_size = 5
train_generator = Data_Generator_Class(record_paths=train_path)
validation_generator = Data_Generator_Class(record_paths=validation_path)
image_processors_train += [
            {'shuffle': len(train_generator)}, {'batch': batch_size}, {'repeat'}]
image_processors_validation += [{'repeat'}]
train_generator.compile_data_set(image_processors_train)
validation_generator.compile_data_set(image_processors_validation)


#%% md

### Alright, lets make our model!

#%%

from Deep_Learning.Easy_VGG16_UNet.Keras_Fine_Tune_VGG16_TF2 import VGG_16
from Deep_Learning.Base_Deeplearning_Code.Visualizing_Model.Visualing_Model import visualization_model_class
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1 import GPUOptions, ConfigProto, Session
from tensorflow.python.keras.backend import set_session
from Deep_Learning.Base_Deeplearning_Code.Callbacks.TF2_Callbacks import MeanDSC


K.clear_session()
gpu_options = GPUOptions(allow_growth=True)
sess = Session(config=ConfigProto(gpu_options=gpu_options, log_device_placement=False))
set_session(sess)
network = {'Layer_0': {'Encoding': [64, 64], 'Decoding': [64]},
           'Layer_1': {'Encoding': [128, 128], 'Decoding': [64]},
           'Layer_2': {'Encoding': [256, 256, 256], 'Decoding': [256]},
           'Layer_3': {'Encoding': [512, 512, 512], 'Decoding': [256]},
           'Layer_4': {'Encoding': [512, 512, 512]}}
VGG_model = VGG_16(network=network, activation='relu',filter_size=(3,3))
VGG_model.make_model()
VGG_model.load_weights()
new_model = VGG_model.created_model
model_path = os.path.join(return_sys_path(),'Models')



new_model.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(), MeanDSC(num_classes=2)])
#%% md

## Freezing pre-trained layers

#%%

def freeze_until_name(model,name):
    set_trainable = False
    for layer in model.layers:
        if layer.name == name:
            set_trainable = True
        layer.trainable = set_trainable
    return model
new_model = freeze_until_name(new_model,'Upsampling0_UNet')


model_name = 'VGG_16_Model'
model_path_out = os.path.join(model_path,'VGG_16_frozen')
if not os.path.exists(model_path_out):
    os.makedirs(model_path_out)

#%%

checkpoint = ModelCheckpoint(os.path.join(model_path_out,'best-model.hdf5'), monitor='val_dice_coef_3D', verbose=1, save_best_only=True,
                              save_weights_only=False, mode='max')

add_images = Add_Images_and_LR(log_dir=model_path_out, validation_data=validation_generator.data_set,
                               number_of_images=len(validation_generator), add_images=True, image_frequency=5,
                               threshold_x=True)
tensorboard = TensorBoard(log_dir=model_path_out)
callbacks = [checkpoint, tensorboard, add_images]

#%% md

### Lets view the model real quick

new_model.fit(train_generator.data_set, epochs=3, callbacks=callbacks, steps_per_epoch=len(train_generator),
              validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
              validation_freq=5)




### First, lets import some necessary functions

#%%

from Deep_Learning.Base_Deeplearning_Code.Models.Keras_Models import my_UNet
from Deep_Learning.Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback import CyclicLR
from tensorflow.python.keras.callbacks import ModelCheckpoint
from functools import partial
from tensorflow.python.keras.optimizers import Adam

#%% md

### Define our convolution and strided blocks, strided is used for downsampling

#%%

activation = {'activation': 'relu'}
kernel = (3,3)
pool_size = (2,2)
#{'channels': x, 'kernel': (3, 3), 'strides': (1, 1),'activation':activation}
conv_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3),
                                        'activation': None, 'strides': (1, 1)}}
pooling_downsampling = {'pooling': {'pooling_type': 'Max',
                                    'pool_size': (2, 2), 'direction': 'Down'}}
pooling_upsampling = {'pooling': {'pool_size': (2, 2), 'direction': 'Up'}}

#%% md

### Our architecture will have 2 main parts in each layer, an 'Encoding' and a 'Decoding' side, 'Encoding' goes down, and 'Decoding' goes up


### We will now create our layer dictionary, this tells our UNet what to look like

### If Pooling is left {} it will perform maxpooling and upsampling with pooling()

#%%

layers_dict = {}
layers_dict['Layer_0'] = {'Encoding': [],
                          'Decoding': [],
                          'Pooling':
                              {'Encoding': [],
                               'Decoding': []
                               }}
layers_dict['Base'] = []
layers_dict['Final_Steps'] = []

#%%

layers_dict['Layer_0'] = {'Encoding': [conv_block(16),activation,conv_block(16),activation],
                          'Decoding': [conv_block(32),activation,conv_block(32),activation],
                          'Pooling':
                              {'Encoding': [pooling_downsampling],
                               'Decoding': [pooling_upsampling]
                               }}
layers_dict['Base'] = [conv_block(32),activation,conv_block(32),activation]
layers_dict['Final_Steps'] = [conv_block(2),{'activation':'softmax'}]

#%%

K.clear_session()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
K.set_session(sess)
new_model = my_UNet(kernel=(3, 3), layers_dict=layers_dict, pool_size=(2, 2), input_size=3, image_size=image_size).created_model

#%% md

### Name your model and define other things! Send a list of strings and it will make a folder path

#%%

model_name = 'My_New_Model'
model_path_out = os.path.join(model_path,model_name)
if not os.path.exists(model_path_out):
    os.makedirs(model_path_out)

#%% md

### Lets look at our model

#%%

from tensorflow.python.keras.callbacks_v1 import TensorBoard

#%%

k = TensorBoard(model_path_out)
k.set_model(new_model)


### Set a learning rate and loss metric, also add any metrics you want to track

#%%

min_lr = 5e-6
max_lr = 1e-3
new_model.compile(Adam(lr=min_lr),loss='categorical_crossentropy', metrics=['accuracy',dice_coef_3D])

#%% md

### This is a checkpoint to save the model if it has the highest dice, also to add images

#%% md

#### We will specify that we want to watch the validation dice, and save the one with the highest value

#%%

monitor = 'val_dice_coef_3D'
mode = 'max'
checkpoint = ModelCheckpoint(os.path.join(model_path_out,'best-model.hdf5'), monitor=monitor, verbose=1, save_best_only=True,
                             save_weights_only=False, save_freq='epoch', mode=mode)

#%% md

#### Next, our tensorboard output will add prediction images


#%% md

#### CyclicLR will allow us to change the learning rate of the model as it runs, and Add_LR_To_Tensorboard will let us view it later

#%%

steps_per_epoch = len(train_generator)//3
step_size_factor = 10

cyclic_lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=steps_per_epoch * step_size_factor, mode='triangular2')
add_lr_to_tensorboard = Add_LR_To_Tensorboard()

#%% md

### Combine all callbacks

#%%

callbacks = [cyclic_lrate, add_lr_to_tensorboard, tensorboard, checkpoint]

#%%

new_model.fit_generator(train_generator,epochs=10, workers=10, max_queue_size=200, validation_data=validation_generator,
                       callbacks=callbacks, steps_per_epoch=steps_per_epoch)

