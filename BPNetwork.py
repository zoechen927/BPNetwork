#!/usr/bin/env python3
# coding=utf-8
import math
import sys
import os
import numpy as np
from PIL import Image
import scipy.io as sio

# activation function
def sigmoid(x):
    return np.array(list(map(lambda i: 1 / (1 + np.exp(-i)), x)))

def get_train_pattern():
    # Return the features and labels of the training set
    current_dir = os.getcwd()
    # current_dir = "/home/lxp/F/developing_folder/intelligence_system/bpneuralnet/"
    # train = sio.loadmat(current_dir + "mnist_train.mat")["mnist_train"]
    train = sio.loadmat(current_dir + "/mnist_train.mat")
    # train_label = sio.loadmat( current_dir + "mnist_train_labels.mat")["mnist_train_labels"]
    train_label = sio.loadmat( current_dir + "/mnist_train_labels.mat")
    
    
    train["mnist_train"]  = np.where(train["mnist_train"] > 180, 1, 0)  # Binarization
    return train["mnist_train"] , train_label["mnist_train_labels"]


# For the test set, the pictures in the folder corresponding to each number are expanded into one-dimensional vectors, which are connected together to form the training set for each number. one number per line
def get_test_pattern():
    # return test set
    base_url = os.getcwd() + "/mnist_test/"
    # base_url = "/home/lxp/F/developing_folder/intelligence_system/bpneuralnet/mnist_test/"
    test_img_pattern = []
    # ten digits
    for i in range(10):
        img_url = os.listdir(base_url + str(i))
        t = []
        for url in img_url:
            img = Image.open(base_url + str(i) + "/" + url) # Open digital pictures in turn
            img = img.convert('1')  # Binarization
            img_array = np.asarray(img, 'i')  # convert to int array
            img_vector = img_array.reshape(
                    img_array.shape[0] * img_array.shape[1])  # expand into a one-dimensional array
            t.append(img_vector) # The same number is expanded into a one-dimensional vector and stored in an array
        test_img_pattern.append(t) # Each number is a line, and the addresses of the ten number arrays are placed here
    return test_img_pattern


class BPNetwork:
    # Neural network class
    def __init__(self, in_count, hiden_count, out_count, in_rate, hiden_rate):
        """

        :param in_count:        number of input layers
        :param hiden_count:     number of hidden layers
        :param out_count:       number of output layers
        :param in_rate:         input layer learning rate
        :param hiden_rate:      hidden layer learning rate
        """
        # The number of nodes in each layer
        self.in_count = in_count
        self.hiden_count = hiden_count
        self.out_count = out_count

        # The weights of the input layer to the hidden layer connection are randomly initialized
        self.w1 = 0.2 * \
                np.random.random((self.in_count, self.hiden_count)) - 0.1

        # The weights of the hidden layer to the output layer connection are randomly initialized
        self.w2 = 0.2 * \
                np.random.random((self.hiden_count, self.out_count)) - 0.1

        # Hidden layer bias vector
        self.hiden_offset = np.zeros(self.hiden_count)
        # Output layer bias vector
        self.out_offset = np.zeros(self.out_count)

        # input layer learning rate
        self.in_rate = in_rate
        # hidden layer learning rate
        self.hiden_rate = hiden_rate

# Each row in the training set is the data of a picture
    def train(self, train_img_pattern, train_label):# input layer, the label of the input layer
        if self.in_count != len(train_img_pattern[0]):# Dimensions of each image
            sys.exit("The input layer dimension is not equal to the sample dimension")
        # for num in range(10):
        # for num in range(10):
        for m in range(1,6,1):#Loop to increase training times
            for i in range(len(train_img_pattern)):

                # generate target vector
                target = [0] * 10
                target[train_label[i][0]] = 1 #train_label[i][0] represents which number it is，target[train_label[i][0]] = 1 represents ideal output is 1

                # forward propagation
                # Hidden layer value is equal to input layer*w1+hidden layer bias
                hiden_value = np.dot(train_img_pattern[i], self.w1) + self.hiden_offset
                hiden_value = sigmoid(hiden_value)

                # Compute the output of the output layer
                out_value = np.dot(hiden_value, self.w2) + self.out_offset
                out_value = sigmoid(out_value)

                # backpropagation
                error = target - out_value
                # Calculate output layer error
                out_error = out_value * (1 - out_value) * error
                # Calculate hidden layer error
                hiden_error = hiden_value * \
                        (1 - hiden_value) * np.dot(self.w2, out_error)

                # Update w2, w2 is a matrix of j rows and k columns, storing the weights from the hidden layer to the output layer
                #
                for k in range(self.out_count):
                    # Update the value of the kth column of w2, and connect all nodes of the hidden layer to the kth node of the output layer
                    # Hidden layer learning rate × input layer error × output value of hidden layer
                    # Hidden layer learning rate × output layer error × output value of hidden layer
                    self.w2[:, k] += self.hiden_rate * out_error[k] * hiden_value

                # Update w1, the weight from the input layer to the hidden layer
                for j in range(self.hiden_count):
                    # learning rate * gradient of hidden layer * value of input layer
                    self.w1[:, j] += self.in_rate * \
                            hiden_error[j] * train_img_pattern[i]

                # update bias vector
                self.out_offset += self.hiden_rate * out_error
                self.hiden_offset += self.in_rate * hiden_error

            # The training cycle was successful

            print("The "+str(m)+" time training succeed")


        # The trained weights and offsets are saved
        sio.savemat(os.getcwd() + "/w1.mat",{'w1':self.w1})
        sio.savemat(os.getcwd() + "/w2.mat",{'w2':self.w2})
        sio.savemat(os.getcwd() + "/hiden_offset.mat",{'hiden_offset':self.hiden_offset})
        sio.savemat(os.getcwd() + "/out_offset.mat",{'out_offset':self.out_offset})

    def test(self, test_img_pattern):
        """
        Test the accuracy of the neural network
        :param test_img_pattern[num][t] represents number num'th t time picture
        :return:
        """
        right = np.zeros(10)
        test_sum = 0
        for num in range(10):  # 10 numbers
            # print("identifying", num)
            num_count = len(test_img_pattern[num]) # The number of test images corresponding to each number
            test_sum += num_count
            for t in range(num_count):  # number num'th t time picture
                hiden_value = np.dot(
                        test_img_pattern[num][t], self.w1) + self.hiden_offset
                hiden_value = sigmoid(hiden_value)
                out_value = np.dot(hiden_value, self.w2) + self.out_offset
                out_value = sigmoid(out_value)
                # print(out_value)
                if np.argmax(out_value) == num: # Index: similarity, the index corresponding to the highest similarity is the predicted value
                    # correct identification
                    right[num] += 1
            print("num%d identification accuracy is%f" % (num, right[num] / num_count))

        # Average recognition rate
        print("Average recognition rate is：", sum(right) / test_sum)
        return np.argmax(out_value)


def run():

    # Neural Network Configuration Parameters
    in_count = 28 * 28
    hiden_count = 6
    out_count = 10
    in_rate = 0.02
    hiden_rate = 0.1
    print("The learning rates of the input layer and the output layer are respectively:"+str(in_rate)+","+str(hiden_rate))
    bpnn = BPNetwork(in_count, hiden_count, out_count, in_rate, hiden_rate)# Create a new BP neural network object

    # read in the training set
    train, train_label = get_train_pattern()

    # read test image
    test_pattern = get_test_pattern()
    bpnn.train(train, train_label) # train if not present
    bpnn.test(test_pattern)

if __name__ == "__main__":
    run()







# def testImage(self):

#     img_name = input("输入要识别的图片（mnist文件夹下图片的地址）\n")
#     base_url = os.getcwd()+"/mnist_test/"
#     img_url = base_url + img_name
#     img = Image.open(img_url)
#     img = img.convert('1')  # 二值化
#     img_array = np.asarray(img, 'i')  # 转化为int数组
#     # 得到图片的特征向量
#     img_v = img_array.reshape(img_array.shape[0] * img_array.shape[1])  # 展开成一维数组

#     W1=sio.loadmat(os.getcwd()+"/w1.mat")
#     W2=sio.loadmat(os.getcwd()+"/w2.mat")
#     Hiden_offset=sio.loadmat(os.getcwd()+"/hiden_offset.mat")
#     Out_offset=sio.loadmat(os.getcwd()+"/out_offset.mat")

#     hiden_value = np.dot(img_v, W1["w1"]) + Hiden_offset["hiden_offset"]
#     hiden_value = sigmoid(hiden_value)
#     out_value = np.dot(hiden_value, W2["w2"]) + Out_offset["out_offset"]
#     out_value = sigmoid(out_value)
#     print("预测值：")
#     print(np.argmax(out_value))