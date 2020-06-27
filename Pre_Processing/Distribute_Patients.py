__author__ = 'Brian M Anderson'
# Created on 10/17/2019

import os
import numpy as np


class Separate_files(object):
    def __init__(self, path):
        self.path = path
        separate_dictionary(separate_by_desc(path),path)
def separate_by_desc(path):
    files = [i for i in os.listdir(path) if i.find('Overall_mask') == 0]
    file_dictionary = {}
    for file in files:
        iteration = file.split('_y')[1].split('.nii')[0]
        desc = file.split('Overall_mask_')[1].split('_y')[0]
        total_desc = desc + iteration
        if total_desc not in file_dictionary:
            file_dictionary[total_desc] = [file]
        else:
            file_dictionary[total_desc].append(file)
        file_dictionary[total_desc].append(file.replace('_y','_').replace('Overall_mask','Overall_Data'))
    return file_dictionary

def separate_dictionary(file_dictionary, path=None):
    patient_image_keys = list(file_dictionary.keys())
    perm = np.arange(len(patient_image_keys))
    np.random.shuffle(perm)
    patient_image_keys = list(np.asarray(patient_image_keys)[perm])
    split_train = int(len(patient_image_keys) / 6)
    for out in ['Train','Test','Validation']:
        if not os.path.exists(os.path.join(path,out)):
            os.makedirs(os.path.join(path,out))
    for xxx in range(split_train):
        for file in file_dictionary[patient_image_keys[xxx]]:
            os.rename(os.path.join(path,file), os.path.join(path,'Test', file))
    for xxx in range(split_train, int(split_train * 2)):
        for file in file_dictionary[patient_image_keys[xxx]]:
            os.rename(os.path.join(path,file), os.path.join(path,'Validation', file))
    for xxx in range(int(split_train * 2), len(perm)):
        for file in file_dictionary[patient_image_keys[xxx]]:
            os.rename(os.path.join(path,file), os.path.join(path,'Train', file))

def main():
    pass


if __name__ == '__main__':
    main()
