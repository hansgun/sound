import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import math
import os

class snd_dns_cal :
    def __init__(self, search_size=21, neighbor_size=13) : 
        #self.wav_file_str = wav_file_str
        self.search_size = search_size
        self.neighbor_size = neighbor_size
        print('\t search_size : ', self.search_size, '\t neighbor_size :', self.neighbor_size)

    def log_specgram(self, audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
        return freqs, np.log(spec.T.astype(np.float32) + eps)


    def get_wave_file(self, wav_file_str) : 
        self.wav_file_str = wav_file_str
        self.samplerate, self.test_sound  = wavfile.read(self.wav_file_str)
        
        self.reshape_snd_data()
        
        return self
    
    
    def reshape_snd_data(self) :
        self.DIMEN_N = int(round(math.sqrt(len(self.test_sound)),0)-1)
        self.DIMEN_N_2 = self.DIMEN_N**2
        #self.test_sound_2d = self.test_sound[:self.DIMEN_N_2].reshape(-1,self.DIMEN_N)
        self.test_sound_2d = self.test_sound[:self.DIMEN_N_2].reshape((self.DIMEN_N,self.DIMEN_N), order='F')
        #print('\n---------------test_sound_2d------------\n')
        #print(self.test_sound_2d)
        #print('\n---------------test_sound_2d------------\n')
        return self
    
    
    ## deprecated
    def normalization_bak(self) : 
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        test_sound_2d_norm = scaler.fit_transform(self.test_sound_2d) * 255
        self.test_sound_2d_norm = test_sound_2d_norm.astype(int)

    ## new normalize function
    def normalization(self) : 
        min_val, max_val = np.min(self.test_sound_2d).astype(np.int32), np.max(self.test_sound_2d).astype(np.int32)
        self.test_sound_2d_norm = ((self.test_sound_2d - min_val) / (max_val - min_val) * 255).astype(int)
        #print('\n------- min, max------- \n')
        #print(min_val,max_val)
        #print('\n------- test_sound_2d_norm------- \n')
        #print(self.test_sound_2d_norm)
        #print('\n------- test_sound_2d_norm------- \n')
        

    def set_snd_data(self, sound_data, samplerate) : 
        '''
        @sound_data ::: sliced sound_data from wav_file 
        '''
        self.test_sound = sound_data
        self.samplerate = samplerate
        #print(self.test_sound)
        self.reshape_snd_data()
        
        return self

                               
    def get_pages(self) :
        '''
        전체 matrix를 계산할 sub-matrix 로 분할하여 그 리스트를 return 하는 함수
        '''
        self.ND_LIST = []
        ## calcluate n*n 의 갯수
        size_x, size_y = self.test_sound_2d.shape
        div_ = self.search_size + self.neighbor_size - 1
        ind_x, ind_y = size_x // div_, size_y // div_
        #cent_p = (div_ -1, div_ -1)

        ## slicing
        for i in range(ind_x) : 
            for j in range(ind_y) : 
                self.ND_LIST.append(self.test_sound_2d_norm[i*div_:(i+1)*div_,j*div_:(j+1)*div_])
        
        #print(self.ND_LIST)
        return self.ND_LIST  

    def ed_dist(self, nd_1, nd_2, round_arg = 1) :
        '''
        2개의 matrix의 ED distance를 return 하는 함수
        round_arg : 소숫점 자리수. default : 1
        '''
        temp_nd = (nd_1 - nd_2).reshape(-1) ** 2
        return round(math.sqrt(sum(temp_nd)),round_arg)

    
    def cal_dns_mat(self) :
        #self.get_wave_file()
        self.normalization()
        self.get_pages()
        '''
        matrix array에 대한 dns 계산하여 np array (SEARCH_SIZE X SEARCH_SIZE X len(ND_LIST)) 를 return
        '''

        ### cent of matrix position 
        CENT_P = (self.search_size//2 + self.neighbor_size//2 , self.search_size//2 + self.neighbor_size//2 )
        
        ### result array
        result_mat = np.array(np.zeros((self.search_size,self.search_size,len(self.ND_LIST))))
        #print(nd_array[0][CENT_P[0]-(neighbor_size//2):CENT_P[0]+(neighbor_size//2)+1,CENT_P[1]-(neighbor_size//2):CENT_P[1]+(neighbor_size//2)+1].shape)
        
        ## for phase
        for ind_mat, mat_x in enumerate(self.ND_LIST) : ## number of nd_array
            single_mat = np.zeros((self.search_size,self.search_size))
            #print('----init....\n',single_mat, '\n\n')
            for x in range(self.search_size) : ## size of x-axis
                for y in range(self.search_size) : ## size of y-axis
                    ## calculate ED distance
                    ## update result of matrix
                    single_mat[x,y] = self.ed_dist(mat_x[x:x+self.neighbor_size,y:y+self.neighbor_size],
                                            mat_x[CENT_P[0]-(self.neighbor_size//2):CENT_P[0]+(self.neighbor_size//2)+1,CENT_P[1]-(self.neighbor_size//2):CENT_P[1]+(self.neighbor_size//2)+1])
            #print('----after update....\n',single_mat, '\n\n')
            ## update center of matrix as np.zeros
            #single_mat[CENT_P,CENT_P] = 0 
            
            #print('----finally....\n',single_mat, '\n\n')
            ## put matrix to final result matrix
            result_mat[:,:,ind_mat] = single_mat
        ## get a mean value of each cell finally
        ##
        ##
        ## return result matrix
        return result_mat.mean(axis=2)

if __name__ == "__main__":
    FILE_PATH = '../data/'
    OUTPUT_PATH = '../output'
    DATA_LOC = 'blues.00000.wav'
    cls_data = snd_dns_cal().get_wave_file(os.path.join(FILE_PATH,DATA_LOC))
    final_mat = cls_data.cal_dns_mat()
    #final_mat = result.mean(axis=2)
    print('\n---------------final_mat------------\n\n')
    print(final_mat)

    plt.imshow(final_mat, cmap='gray')
    plt.savefig(os.path.join(OUTPUT_PATH,'fig1.png'), dpi=300)