# _*_coding:utf-8 _*_
# @Time    :2021/10/8
# @Author  :Baiyun Chen
# @FileName: Self_adaptive_relative_density.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import random
from collections import Counter
import math
from sklearn.neighbors import NearestNeighbors as NNs
from sklearn.cluster import k_means
from sklearn.mixture import GaussianMixture



def load_data(fname):
    with open(fname, 'rt') as csvfile:
        data = pd.read_csv(csvfile, header=None)
    return data


def split_data_set(data, proportion, random_state=2):
    """
    split the data into two parts based on the given proportion
    :param data: ndarray
    :param proportion: eg:proportion = [0.8, 0.2], the sum of the parameters should be one
    :param random_state:
    :return: the splitted data set, eg: containing 80% for training and 20% for testing
    """
    assert sum(proportion) == 1
    random.seed(random_state)
    n = data.shape[0] #row number
    index = list(range(n))
    random.shuffle(index)
    data_set = []
    count_n = 0
    for i in range(len(proportion)):
        n = int(sum(proportion[:i + 1]) * data.shape[0])
        data_set.append(data[index[count_n: n], :])
        count_n = count_n + n
    return data_set

def add_flip_noise(dataset, noise_rate):
    """
    :param dataset: training data, ndarray
    :param noise_rate: the given rate to flip the labels in each class
    :return: dataset with flipped data sets, it contains a flag(the last column) to indicate whether the label of this samples has been flipped
    """
    samples_num = dataset.shape[0]
    p_label, n_label = max(dataset[:, 0]), min(dataset[:, 0])
    positive_noise = int(len(dataset[dataset[:, 0] == p_label]) * noise_rate[0])
    negative_noise = int(len(dataset[dataset[:, 0] == n_label]) * noise_rate[1])

    noise_index_list = []
    p_index = 0
    n_index = 0
    np.random.seed()

    # add flag to record the flipped labels
    F = np.zeros((len(dataset)))

    while True:
        rand_index = int(np.random.uniform(0, samples_num))
        if rand_index in noise_index_list:
            continue

        if dataset[rand_index, 0] == p_label and p_index < positive_noise:
            dataset[rand_index, 0] = n_label
            p_index += 1
            noise_index_list.append(rand_index)

        elif dataset[rand_index, 0] == n_label and n_index < negative_noise:
            dataset[rand_index, 0] = p_label
            n_index += 1
            noise_index_list.append(rand_index)
        if p_index >= positive_noise and n_index >= negative_noise:
            break

    for i in range(len(dataset)):
        if i in noise_index_list:
            F[i] = 1
        else:
            F[i] = 0

    dataset_Flag = np.c_[dataset, F]
    return dataset_Flag


def cal_euclidean_distance(a, b):
    return np.sqrt(sum(np.power((a - b), 2)))

class Self_adaptive_RD:
    def __init__(self, data, random_state=None):
        """
        :param data: array for all data with label in the first column.
        :param k: Number of nearest neighbors.
        """
        self.data = data
        self.div_data()
        self.n, self.m = data.shape


    def div_data(self):
        """"
        divide the dataset into pos and neg, resort the train data with new index
        return: the divided pos data, neg data, and train data
        """
        count = Counter(self.data[:, 0])
        a, b = set(count.keys())
        self.pos_label, self.neg_label = (a, b) if a > b else (b, a)
        self.pos_data = self.data[self.data[:, 0] == self.pos_label]
        self.neg_data = self.data[self.data[:, 0] == self.neg_label]

        self.train = np.vstack((self.pos_data, self.neg_data))
        return self.pos_data, self.neg_data, self.train

    def calculate_neighbors_Euclidean(self, data1, data2, k):
        """
        Utilize Euclidean distance to calculate the neighbors for each sample
        :param k: the number of nearest neighbors
        :return: the distance of homogeneous neighbors and heterogeneous neighbors for each sample in each class
        """
        # when calculating the homogeneous neighbors, the number of neighbors should add 1 to exclude the sample itself
        pos_distance_k, pos_index_k = NNs(n_neighbors=k + 1, metric=cal_euclidean_distance).fit(data1).kneighbors(data1,
                                                                                                                  return_distance=True)

        neg_distance_k, neg_index_k = NNs(n_neighbors=k + 1, metric=cal_euclidean_distance).fit(data2).kneighbors(data2,
                                                                                                                  return_distance=True)
        pos_heter_distance_k, pos_heter_index_k = NNs(n_neighbors=k, metric=cal_euclidean_distance).fit(
            data2).kneighbors(
            data1, return_distance=True)
        neg_heter_distance_k, neg_heter_index_k = NNs(n_neighbors=k, metric=cal_euclidean_distance).fit(
            data1).kneighbors(
            data2, return_distance=True)
        return pos_distance_k, neg_distance_k, pos_heter_distance_k, neg_heter_distance_k

    def calculate_p_relative_density(self, pos_distance_k, neg_distance_k, pos_heter_distance_k, neg_heter_distance_k,p=1):
        """
        :param pos_distance_k:
        :param neg_distance_k:
        :param pos_heter_distance_k:
        :param neg_heter_distance_k:
        :param p: the power of basic relative density to help amplify or narrow the gaps between noisy samples and normal samples
        :return: the calculated p_relative_density value of each sample in each class
        """
        pos_distance_k_sum = np.sum(pos_distance_k, axis=1)
        pos_heter_distance_k_sum = np.sum(pos_heter_distance_k, axis=1)
        pos_relative_density = np.divide(pos_distance_k_sum,
                                         pos_heter_distance_k_sum if pos_heter_distance_k_sum.all() != 0 else -1)

        pos_relative_density= [math.pow(prd,p) for prd in pos_relative_density]

        neg_distance_k_sum = np.sum(neg_distance_k, axis=1)
        neg_heter_distance_k_sum = np.sum(neg_heter_distance_k, axis=1)
        neg_relative_density = np.divide(neg_distance_k_sum,
                                         neg_heter_distance_k_sum if neg_heter_distance_k_sum.all() != 0 else -1)

        neg_relative_density = [math.pow(nrd, p) for nrd in neg_relative_density]

        pos_relative_density = np.array(list(map(lambda x: min(1000, x), pos_relative_density)))
        neg_relative_density = np.array(list(map(lambda x: min(1000, x), neg_relative_density)))

        return pos_relative_density, neg_relative_density

    def check_noise_RD(self, train, pos_relative_density, neg_relative_density):
        """
        employ groupby function to check whether there exists difference of relative density values between noisy and normal samples
        """
        train_relative_density = np.hstack((pos_relative_density, neg_relative_density))
        data_all = np.c_[train, train_relative_density]
        check_noise_data = np.c_[data_all[:, 0], data_all[:, -2:]]

        check_noise_data_pd = pd.DataFrame(check_noise_data)
        check_noise_data_pd.columns = ["Label", "Flag", "RD"]

        df_noise= check_noise_data_pd.groupby(['Label', 'Flag']).agg([np.mean, np.std, np.max, np.min])
        #将行列标签的结果保存至df
        df_noise_flattern=pd.DataFrame(df_noise.to_records())
        df_noise_flattern.columns=[
        column.replace("('RD',", "").replace(")", "").replace("'", "").replace(" ", "") for column in
        df_noise_flattern.columns]
        return df_noise_flattern,check_noise_data_pd

    def k_means_ratio(self, relative_density, k_clusters=2):
        """
        employ 2-means cluster the relative density value of each class to search for the threshold dividedly
        """
        irt = 10
        truncate_percentage = 0.95
        wt_array = []
        relative_density_reshape = relative_density.reshape(len(relative_density), 1)
        for i in range(irt):
            rd_clusters = k_means(X=relative_density_reshape, n_clusters=k_clusters, init='random', n_init=1,max_iter=300)
            rd_cluster_labels = pd.Series(rd_clusters[1])
            rd_cluster_centers = rd_clusters[0]
            rd_cluster_count = rd_cluster_labels.value_counts()

            if len(rd_cluster_centers) > 1:
                rd_cluster_0 = []
                rd_cluster_1 = []
                if rd_cluster_centers[0] < rd_cluster_centers[1]:
                    rd_cluster_0.append(pd.DataFrame(relative_density).iloc[rd_cluster_labels.values == 0])
                    rd_cluster_1.append(pd.DataFrame(relative_density).iloc[rd_cluster_labels.values == 1])
                    weight = rd_cluster_count[1] / len(relative_density)

                else:
                    rd_cluster_0.append(pd.DataFrame(relative_density).iloc[rd_cluster_labels.values == 1])
                    rd_cluster_1.append(pd.DataFrame(relative_density).iloc[rd_cluster_labels.values == 0])
                    weight = rd_cluster_count[0] / len(relative_density)
            else:

                count = Counter(rd_cluster_labels)
                weight = min(count.values()) / len(rd_cluster_labels)

            relative_density = relative_density.reshape(1, -1)[0]
            #truncate the relative density with truncate_percentage to avoid the potential outliers
            threshold_location = int(len(relative_density) * weight * truncate_percentage)
            #threshold: the minmum value of the truncated cluster with bigger center value
            threshold = relative_density[np.argsort(-relative_density)[threshold_location - 1]]

            # repeat clustering process ten time to reduce the randomness aroused by the arbitrarily selected initial points
            wt_array_row = []
            wt_array_row.append(weight)
            wt_array_row.append(threshold)
            wt_array.append(wt_array_row)
        wt_array=np.asarray(wt_array)

        threshold=np.max(wt_array[:,1])
        weight=len(list(filter(lambda x: x >= threshold, relative_density))) / len(relative_density)

        return weight, threshold


    def gmm_ratio(self, relative_density, n_components=2):
        """
        employ GMM cluster the relative density value of each class to search for the threshold dividedly
        """
        irt = 10
        truncate_percentage=0.95
        wt_array = []
        relative_density_reshape = relative_density.reshape(len(relative_density), 1)
        for i in range(irt):
            gm = GaussianMixture(n_components=n_components,init_params='random',n_init=1,max_iter=100).fit(relative_density_reshape)
            means = gm.means_
            weights = gm.weights_

            if means.shape[1]>1:
                weights = weights[:, np.newaxis]  # 行向量换成列向量
                means_weights = np.c_[means, weights]
                idx = np.lexsort(means_weights[:, ::-1].T)
                weight = weights[idx[-1]]
            else:
                weight = min(weights)

            relative_density = relative_density.reshape(1,-1)[0]

            # truncate the relative density with percentage to avoid the potential outliers
            threshold_location = int(len(relative_density) * weight * truncate_percentage)
            threshold = relative_density[np.argsort(-relative_density)[threshold_location-1]]

            wt_array_row = []
            wt_array_row.append(weight)
            wt_array_row.append(threshold)
            wt_array.append(wt_array_row)
        wt_array = np.asarray(wt_array)

        threshold = np.max(wt_array[:, 1])
        weight = len(list(filter(lambda x: x >= threshold, relative_density))) / len(relative_density)

        return weight,threshold

    def filter_noise(self, data, relative_density, rd_threshold):
        """

        :param data:
        :param relative_density:
        :param rd_threshold:
        :return: the filtered clean data, and the idt_flag to mark whether the sample is identified as a label noise
        """

        Idt_Flag = np.ones((len(relative_density)))
        data_rd = np.c_[data, relative_density]
        idx_clean=np.where(relative_density<rd_threshold)[0]
        data_clean=data_rd[idx_clean,:-1]
        for i in idx_clean:
            Idt_Flag[i] = 0

        return data_clean, Idt_Flag


def main():
    data = load_data(r'D:\Exp\DataSet\fourclass.csv')

    data_ndarray = data.values
    train_set_ori, test_set = split_data_set(data_ndarray, [0.8, 0.2])

    train_set = add_flip_noise(train_set_ori, [0.3, 0.3])  # 最后一列增加了Flag标识样本是否为翻转噪声
    pos_data, neg_data, train = Self_adaptive_RD(train_set).div_data()

    pos_distance_k_Euclidean, neg_distance_k_Euclidean, pos_heter_distance_k_Euclidean, neg_heter_distance_k_Euclidean = Self_adaptive_RD(
        train_set).calculate_neighbors_Euclidean(pos_data[:, 1:-1], neg_data[:, 1:-1], 7)
    pos_relative_density_Euclidean, neg_relative_density_Euclidean = Self_adaptive_RD(
        train_set).calculate_p_relative_density(pos_distance_k_Euclidean, neg_distance_k_Euclidean,
                                              pos_heter_distance_k_Euclidean, neg_heter_distance_k_Euclidean)


    # set_threshold_k_means
    pos_k_means_weight,pos_k_means_threshold = Self_adaptive_RD(train_set).k_means_ratio(
        pos_relative_density_Euclidean)
    pos_k_means_data_filtered,pos_k_means_Idt_Flag = Self_adaptive_RD(train_set).filter_noise(pos_data[:, 1:-1],
                                                                                 pos_relative_density_Euclidean,
                                                                                 pos_k_means_threshold)

    neg_k_means_weight,neg_k_means_threshold = Self_adaptive_RD(train_set).k_means_ratio(
        neg_relative_density_Euclidean)
    neg_k_means_data_filtered,neg_k_means_Idt_Flag = Self_adaptive_RD(train_set).filter_noise(neg_data[:, 1:-1],
                                                                                 neg_relative_density_Euclidean,
                                                                                 neg_k_means_threshold)
    clean_data_k_means=np.vstack((pos_k_means_data_filtered,neg_k_means_data_filtered))

    print("pos_k_means_threshold:",pos_k_means_threshold)
    print("neg_k_means_threshold:", neg_k_means_threshold)
    print("clean_data_k_means:", clean_data_k_means.shape)

    #set_threshold_GMM:
    pos_gmm_weight, pos_gmm_threshold=Self_adaptive_RD(train_set).gmm_ratio(pos_relative_density_Euclidean)
    pos_gmm_data_filtered, pos_gmm_Idt_Flag = Self_adaptive_RD(train_set).filter_noise(pos_data[:, 1:-1],
                                                                                       pos_relative_density_Euclidean, pos_gmm_threshold)

    neg_gmm_weight,neg_gmm_threshold = Self_adaptive_RD(train_set).gmm_ratio(
        neg_relative_density_Euclidean)
    neg_gmm_data_filtered,neg_gmm_Idt_Flag = Self_adaptive_RD(train_set).filter_noise(neg_data[:, 1:-1],
                                                                                 neg_relative_density_Euclidean,
                                                                                 neg_gmm_threshold)
    clean_data_gmm=np.vstack((pos_gmm_data_filtered,neg_gmm_data_filtered))
    print("pos_gmm_threshold:", pos_gmm_threshold)
    print("neg_gmm_threshold:", neg_gmm_threshold)
    print("clean_data_gmm:", clean_data_gmm.shape)

if __name__ == '__main__':
    main()
