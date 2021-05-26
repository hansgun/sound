import os

import numpy as np
from numpy import linalg as LA

NEIGHBOR_SIZE = 13
SEARCH_SIZE = 21


def normalization(test_sound_2d):
    min_val, max_val = np.min(test_sound_2d).astype(np.int32), np.max(test_sound_2d).astype(np.int32)
    return ((test_sound_2d - min_val) / (max_val - min_val) * 255).astype(int)


def reshape_snd_data(test_sound_2d_norm):
    import math
    DIMEN_N = int(round(math.sqrt(test_sound_2d_norm.shape[1]), 0) - 1)
    DIMEN_N_2 = DIMEN_N ** 2
    ND_LIST = []
    # self.test_sound_2d = self.test_sound[:self.DIMEN_N_2].reshape(-1,self.DIMEN_N)
    for x in test_sound_2d_norm:
        ND_LIST.append(x[:DIMEN_N_2].reshape((DIMEN_N, DIMEN_N), order='F'))
    return ND_LIST


def get_pages(test_sound_2d, search_size=SEARCH_SIZE, neighbor_size=NEIGHBOR_SIZE):
    """
    전체 matrix를 계산할 sub-matrix 로 분할하여 그 리스트를 return 하는 함수
    """
    ND_LIST = []

    # calcluate n*n 의 갯수
    size_x, size_y = test_sound_2d[0].shape
    div_ = search_size + neighbor_size - 1
    ind_x, ind_y = size_x // div_, size_y // div_
    # cent_p = (div_ -1, div_ -1)
    result_list = []

    # slicing
    for x in test_sound_2d:
        for i in range(ind_x):
            for j in range(ind_y):
                ND_LIST.append(x[i * div_:(i + 1) * div_, j * div_:(j + 1) * div_])
        result_list.append(ND_LIST)
        ND_LIST = []
    # print(self.ND_LIST)
    return np.asarray(result_list)


def distance_matrix(mat_x, search_win, search_size, neighbor_size):
    return np.array([LA.norm(
        mat_x[x:x + neighbor_size, y:y + neighbor_size] - search_win
    ) for x in range(search_size)
        for y in range(search_size)
    ]
    ).reshape(search_size, search_size, 1)


def cal_dns_mat(sliced_array, per_length, search_size=SEARCH_SIZE, neighbor_size=NEIGHBOR_SIZE):
    """
    matrix array에 대한 dns 계산하여 np array (SEARCH_SIZE X SEARCH_SIZE X len(ND_LIST)) 를 return
    """

    # cent of matrix position
    CENT_P = (search_size // 2 + neighbor_size // 2, search_size // 2 + neighbor_size // 2)

    # result array
    result_mat = np.array([])
    return_result = np.zeros((search_size, search_size, sliced_array.shape[0]))
    # for phase
    for ind_mat, x in enumerate(sliced_array):  # number of nd_array
        # if ind_mat % 100 == 0 : print('{} of {}'.format(ind_mat,len(paged_norm_sliced)))
        for ind_inner, mat_x in enumerate(x):
            search_win = mat_x[CENT_P[0] - (neighbor_size // 2):CENT_P[0] + (neighbor_size // 2) + 1,
                         CENT_P[1] - (neighbor_size // 2):CENT_P[1] + (neighbor_size // 2) + 1].copy()
            if ind_inner == 0:
                result_mat = distance_matrix(mat_x, search_win, search_size, neighbor_size)
            else:
                result_mat = np.concatenate(
                    (result_mat, distance_matrix(mat_x, search_win, search_size, neighbor_size)), axis=2)

        return_result[:, :, ind_mat] = np.mean(result_mat, axis=2)
    # get a mean value of each cell finally
    return return_result


if __name__ == "__main__":
    import time
    import sys
    import snd_loader

    HOME_PATH = '/Users/hansgun/Documents/code/sound'
    DATA_DIR = 'data'
    MODULE_DIR = 'farms'
    sys.path.append(os.path.join(HOME_PATH, MODULE_DIR))

    # search_size=21, neighbor_size=13
    NEIGHBOR_SIZE = 13
    SEARCH_SIZE = 21

    start_time = time.time()
    wav_file_str = os.path.join(HOME_PATH, DATA_DIR, 'real.wav')
    print(wav_file_str)

    samplerate, sliced = snd_loader.snd_loader(wav_file_str, 3, 1).get_snd_df()

    norm_sliced = normalization(sliced)

    reshaped = reshape_snd_data(np.array(sliced))

    paged_norm_sliced = get_pages(reshaped)

    result = cal_dns_mat(np.asarray(paged_norm_sliced), per_length=len(paged_norm_sliced[0]))

    print('elapsed : ', time.time() - start_time)
