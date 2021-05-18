import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import math
import os
from numpy import linalg as LA
import sys 

class snd_dns_cal :
    def __init__(self, search_size=21, neighbor_size=13) : 
        #self.wav_file_str = wav_file_str
        self.search_size = search_size
        self.neighbor_size = neighbor_size
        #print('dns_cal ::', '\t search_size : ', self.search_size, '\t neighbor_size :', self.neighbor_size)

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

    def get_pages_bak(self) :
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
        #print(np.sum(np.sqrt((nd_1-nd_2)**2)))
        #return round(np.sqrt(np.sum((nd_1-nd_2)**2)),round_arg)

    
    def cal_dns_mat_bak(self) :
        #self.get_wave_file()
        self.normalization()
        self.get_pages()
        '''
        matrix array에 대한 dns 계산하여 np array (SEARCH_SIZE X SEARCH_SIZE X len(ND_LIST)) 를 return
        '''

        ### cent of matrix position 
        CENT_P = (self.search_size//2 + self.neighbor_size//2 , self.search_size//2 + self.neighbor_size//2 )
        
        ### result array
        result_mat = np.zeros((self.search_size,self.search_size))
        #print(nd_array[0][CENT_P[0]-(neighbor_size//2):CENT_P[0]+(neighbor_size//2)+1,CENT_P[1]-(neighbor_size//2):CENT_P[1]+(neighbor_size//2)+1].shape)
        
        ## for phase
        for ind_mat, mat_x in enumerate(self.ND_LIST) : ## number of nd_array
            single_mat = np.zeros((self.search_size,self.search_size))
            #print('----init....\n',single_mat, '\n\n')
            for x in range(self.search_size) : ## size of x-axis
                for y in range(self.search_size) : ## size of y-axis
                    ## calculate ED distance
                    ## update result of matrix
                    single_mat[x,y] = LA.norm(mat_x[x:x+self.neighbor_size,y:y+self.neighbor_size] - mat_x[CENT_P[0]-(self.neighbor_size//2):CENT_P[0]+(self.neighbor_size//2)+1,CENT_P[1]-(self.neighbor_size//2):CENT_P[1]+(self.neighbor_size//2)+1])
            
            #single_mat = np.array([LA.norm(mat_x[x:x+self.neighbor_size,y:y+self.neighbor_size] - 
            #                               mat_x[CENT_P[0]-(self.neighbor_size//2):CENT_P[0]+(self.neighbor_size//2)+1,CENT_P[1]-(self.neighbor_size//2):CENT_P[1]+(self.neighbor_size//2)+1]) 
            #                       for y in range(self.search_size) for x in range(self.search_size)]).reshape((self.search_size,self.search_size), order='F')
            #print('----after update....\n',single_mat, '\n\n')
            ## update center of matrix as np.zeros
            #single_mat[CENT_P,CENT_P] = 0 
            
            #print('----finally....\n',single_mat, '\n\n')
            ## put matrix to final result matrix
            result_mat = result_mat + single_mat
        ## get a mean value of each cell finally
        ##
        ##
        ## return result matrix
        return result_mat / len(self.ND_LIST)

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
        result_mat = np.zeros((self.search_size,self.search_size, len(self.ND_LIST)))
        
        ## for phase
        for ind_mat, mat_x in enumerate(self.ND_LIST) : ## number of nd_array
            search_win = mat_x[CENT_P[0]-(self.neighbor_size//2):CENT_P[0]+(self.neighbor_size//2)+1,
                               CENT_P[1]-(self.neighbor_size//2):CENT_P[1]+(self.neighbor_size//2)+1]
                               
            result_mat[:,:,ind_mat] = np.array([ LA.norm(
                                                    mat_x[x:x+self.neighbor_size,y:y+self.neighbor_size] - search_win
                                                         ) for x in range(self.search_size) 
                                                         for y in range(self.search_size)]
                                               ).reshape(self.search_size, -1)
        ## get a mean value of each cell finally
        #return result_mat / len(self.ND_LIST)
        return np.mean(result_mat, axis=2)
    
def main(argv):
    import  getopt
    FILE_NAME     = argv[0] # command line arguments의 첫번째는 파일명
    MULTI_PROCESS = 1     # num of process

    try:
        # opts: getopt 옵션에 따라 파싱 ex) [('-i', 'myinstancce1')]
        # etc_args: getopt 옵션 이외에 입력된 일반 Argument
        # argv 첫번째(index:0)는 파일명, 두번째(index:1)부터 Arguments
        opts, etc_args = getopt.getopt(argv[1:], \
                                 "hp:", ["help","multi_process="])

    except getopt.GetoptError: # 옵션지정이 올바르지 않은 경우
        print(FILE_NAME, '-p <num_of_process>')
        sys.exit(2)

    for opt, arg in opts: # 옵션이 파싱된 경우
        if opt in ("-h", "--help"): # HELP 요청인 경우 사용법 출력
            print(FILE_NAME, '-p <num_of_process>')
            sys.exit()

        elif opt in ("-p", "--multi_process"):
            MULTI_PROCESS = int(arg)



    #if MULTI_PROCESS > 0 & MULTI_PROCESS <= 1: # 필수항목 값이 비어있다면
    #    print(FILE_NAME, "-p <num_of_process>") # 필수임을 출력
    #    sys.exit(2)

    print("MULTI_PROCESS:", MULTI_PROCESS)
    
    
    return MULTI_PROCESS

if __name__ == "__main__":
    ##MULTI_PROCESS = main(sys.argv)
    
    import time
    
    #HOME_PATH = '/Users/han/Documents/code/python/sound/'
    HOME_PATH = 'C:\\hansgun\\sound'
    FILE_PATH = 'data/'
    OUTPUT_PATH = 'output'
    DATA_LOC = 'real.wav'
    MODULE_DIR = 'farms'
    
    import sys
    sys.path.append(os.path.join(HOME_PATH,MODULE_DIR))
    import snd_loader
    NEIGHBOR_SIZE , SEARCH_SIZE = (13, 21)
    start_time = time.time()
    
    
    ### test for multi-thread
    MULTI_PROCESS = 1
    
    samplerate, sliced = snd_loader.snd_loader(os.path.join(HOME_PATH,FILE_PATH,DATA_LOC), 3, 1).get_snd_df()
    final_mat = np.zeros((SEARCH_SIZE,SEARCH_SIZE,len(sliced)))
    
    
    #### single thread
    if MULTI_PROCESS <= 1 : 
        print('single')
        #cls_data = snd_dns_cal().get_wave_file(os.path.join(HOME_PATH,FILE_PATH,DATA_LOC))
        #final_mat = cls_data.cal_dns_mat()
        
        for i, x in enumerate(sliced) :
            print(i)
            final_mat[:,:,i] =  snd_dns_cal(neighbor_size=NEIGHBOR_SIZE, search_size=SEARCH_SIZE).set_snd_data(x,samplerate).cal_dns_mat()
    
    ## multi thread
    else :
        print('multi :', MULTI_PROCESS)
        import concurrent.futures
        
        
        
        def cal_final_map(sliced) :
            return snd_dns_cal(neighbor_size=NEIGHBOR_SIZE, search_size=SEARCH_SIZE).set_snd_data(sliced,samplerate).cal_dns_mat()

        #with concurrent.futures.ProcessPoolExecutor(max_workers=MULTI_PROCESS) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MULTI_PROCESS) as executor:
            for number, node in zip(range(len(sliced)),executor.map(cal_final_map, sliced)):
                final_mat[:,:,number] = node
                print(number)

        '''
        def cal_final_map(sliced, i, samplerate) :
            return (snd_dns_cal(neighbor_size=NEIGHBOR_SIZE, search_size=SEARCH_SIZE).set_snd_data(sliced,samplerate).cal_dns_mat())
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MULTI_PROCESS) as executor:
        #with concurrent.futures.ThreadPoolExecutor(max_workers=MULTI_PROCESS) as executor:
            future_to_list = {executor.submit(cal_final_map, x, i, samplerate):i for i, x in enumerate(sliced)}
            for j,future in enumerate(concurrent.futures.as_completed(future_to_list)):
                url = future_to_list[future]
                print(j)
                try:
                    final_mat[:,:,j] = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (j, exc))
        '''
            
           

        #import pickle
        #with open(os.path.join(HOME_PATH,OUTPUT_PATH,'data_680.pickle'), 'wb') as f:
        #     pickle.dump(final_mat, f, pickle.HIGHEST_PROTOCOL)

    print('execution time :', time.time() - start_time)
    #final_mat = result.mean(axis=2)
    print('\n---------------final_mat------------\n\n')
    #print(final_mat[:,:,1])
    plt.imshow(final_mat[:,:,10], cmap='gray')
    #plt.savefig(os.path.join(HOME_PATH,OUTPUT_PATH,'fig1.png'), dpi=300)