#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:54:34 2021

@author: han
"""

import os
from scipy.io import wavfile

import os
from scipy.io import wavfile


class snd_loader:
    def __init__(self, file_path, len_snd, interval_snd):
        self.file_path = file_path
        try:
            self.samplerate, self.test_sound = wavfile.read(self.file_path)
        except Exception:
            print('Error::: not a vaild file path')

        self.len_snd = len_snd
        self.interval_snd = interval_snd

    def get_snd_df(self):
        # print(self.test_sound)
        ind_num = int((len(self.test_sound) // self.samplerate - self.len_snd) // self.interval_snd)

        self.sliced = list([self.test_sound[i * self.interval_snd * self.samplerate:(
                                                                                                i * self.interval_snd + self.len_snd) * self.samplerate,
                            0] for i in range(ind_num)])
        print('dns layer size :', len(self.sliced))
        return (self.samplerate, self.sliced)

if __name__ == "__main__":
    SND_FILE_PATH = os.path.join(os.getcwd(), os.pardir)
    #SND_FILE_PATH = '/Users/han/Documents/code/python/sound/from_source/'
    FILE_NAME = 'real.wav'
    print(os.path.join(SND_FILE_PATH,FILE_NAME))
    samplerate, result_list = snd_loader(os.path.join(SND_FILE_PATH,FILE_NAME), 3, 2).get_snd_df()
    
    print(list(result_list[0]))
    
    import sys
    sys.path.append('/Users/han/Documents/code/python/sound/farms')
    import snd_dns_cal    
    test = snd_dns_cal.snd_dns_cal().set_snd_data(result_list[3],samplerate).cal_dns_mat()
    print(test)