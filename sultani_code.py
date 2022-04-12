from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation ,LSTM,Reshape
from keras.regularizers import l2
from keras.optimizers import adam, Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
import ConfigParser
import collections
import time
import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
import path
from os.path import basename
import glob
import theano.sandbox
theano.sandbox.cuda.use('gpu0')


print("Create Model")
model = Sequential()
model.add(Dense(512, input_dim=4096,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32,init='glorot_normal',W_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(1,init='glorot_normal',W_regularizer=l2(0.001),activation='sigmoid'))


def load_model(json_path): # Function to load the model
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path): # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

def save_model(model, json_path, weight_path): # Function to save the model
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)





# Load Training Dataset

def load_dataset_Train_batch(AbnormalPath, NormalPath):
#    print("Loading training batch")

    batchsize=60       # Each batch contain 60 videos.
    n_exp=batchsize/2  # Number of abnormal and normal videos

    Num_abnormal = 810  # Total number of abnormal videos in Training Dataset.
    Num_Normal = 800    # Total number of Normal videos in Training Dataset.


    # We assume the features of abnormal videos and normal videos are located in two different folders.
    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal-n_exp:] # Indexes for randomly selected Abnormal Videos
    Norm_list_iter = np.random.permutation(Num_Normal)
    Norm_list_iter = Norm_list_iter[Num_Normal-n_exp:]     # Indexes for randomly selected Normal Videos


    AllVideos_Path = AbnormalPath
    def listdir_nohidden(AllVideos_Path):  # To ignore hidden files
        file_dir_extension = os.path.join(AllVideos_Path, '*_C.txt')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)

    All_Videos=sorted(listdir_nohidden(AllVideos_Path))
    All_Videos.sort()
    AllFeatures = []  # To store C3D features of a batch
    print("Loading Abnormal videos Features...")

    Video_count=-1
    for iv in Abnor_list_iter:
        Video_count=Video_count+1
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        num_feat = len(words) / 4096
        # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
        # we have already computed C3D features for the whole video and divide the video features into 32 segments. Please see Save_C3DFeatures_32Segments.m as well

        count = -1;
        VideoFeatues = []
        for feat in xrange(0, num_feat):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))

        if Video_count == 0:
            AllFeatures = VideoFeatues
        if Video_count > 0:
            AllFeatures = np.vstack((AllFeatures, VideoFeatues))
        print(" Abnormal Features  loaded")

        
        
    print("Loading Normal videos...")
    AllVideos_Path =  NormalPath

    def listdir_nohidden(AllVideos_Path):  # To ignore hidden files
        file_dir_extension = os.path.join(AllVideos_Path, '*_C.txt')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)

    All_Videos = sorted(listdir_nohidden(AllVideos_Path))
    All_Videos.sort()

    for iv in Norm_list_iter:
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        feat_row1 = np.array([])
        num_feat = len(words) /4096   # Number of features to be loaded. In our case num_feat=32, as we divide the video into 32 segments.

        count = -1;
        VideoFeatues = []
        for feat in xrange(0, num_feat):


            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))
            feat_row1 = []
        AllFeatures = np.vstack((AllFeatures, VideoFeatues))

    print("Features  loaded")


    AllLabels = np.zeros(32*batchsize, dtype='uint8')
    th_loop1=n_exp*32
    th_loop2=n_exp*32-1



    for iv in xrange(0, 32*batchsize):
            if iv< th_loop1:
                AllLabels[iv] = int(0)  # All instances of abnormal videos are labeled 0.  This will be used in custom_objective to keep track of normal and abnormal videos indexes.
            if iv > th_loop2:
                AllLabels[iv] = int(1)   # All instances of Normal videos are labeled 1. This will be used in custom_objective to keep track of normal and abnormal videos indexes.
           # print("ALLabels  loaded")

    return  AllFeatures,AllLabels


def custom_objective(y_true, y_pred):
    'Custom Objective function'

    y_true = T.flatten(y_true)
    y_pred = T.flatten(y_pred)

    n_seg = 32  # Because we have 32 segments per video.
    nvid = 60
    n_exp = nvid / 2
    Num_d=32*nvid


    sub_max = T.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
    sub_sum_labels = T.ones_like(y_true) # It is used to sum the labels in order to distinguish between normal and abnormal videos.
    sub_sum_l1=T.ones_like(y_true)  # For holding the concatenation of summation of scores in the bag.
    sub_l2 = T.ones_like(y_true) # For holding the concatenation of L2 of score in the bag.

    for ii in xrange(0, nvid, 1):
        # For Labels
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = T.concatenate([sub_sum_labels, T.stack(T.sum(mm))])  # Just to keep track of abnormal and normal vidoes

        # For Features scores
        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = T.concatenate([sub_max, T.stack(T.max(Feat_Score))])         # Keep the maximum score of scores of all instances in a Bag (video)
        sub_sum_l1 = T.concatenate([sub_sum_l1, T.stack(T.sum(Feat_Score))])   # Keep the sum of scores of all instances in a Bag (video)

        z1 = T.ones_like(Feat_Score)
        z2 = T.concatenate([z1, Feat_Score])
        z3 = T.concatenate([Feat_Score, z1])
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = T.sum(T.sqr(z))
        sub_l2 = T.concatenate([sub_l2, T.stack(z)])


    # sub_max[Num_d:] means include all elements after Num_d.
    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[4:]
    #[  6.  12.   7.  18.   9.  14.]

    sub_score = sub_max[Num_d:]  # We need this step since we have used T.ones_like
    F_labels = sub_sum_labels[Num_d:] # We need this step since we have used T.ones_like
    #  F_labels contains integer 32 for normal video and 0 for abnormal videos. This because of labeling done at the end of "load_dataset_Train_batch"



    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[:4]
    # [ 2 4 3 9]... This shows 0 to 3 elements

    sub_sum_l1 = sub_sum_l1[Num_d:] # We need this step since we have used T.ones_like
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]         # We need this step since we have used T.ones_like
    sub_l2 = sub_l2[:n_exp]


    indx_nor = theano.tensor.eq(F_labels, 32).nonzero()[0]  # Index of normal videos: Since we labeled 1 for each of 32 segments of normal videos F_labels=32 for normal video
    indx_abn = theano.tensor.eq(F_labels, 0).nonzero()[0]

    n_Nor=n_exp

    Sub_Nor = sub_score[indx_nor] # Maximum Score for each of abnormal video
    Sub_Abn = sub_score[indx_abn] # Maximum Score for each of normal video

    z = T.ones_like(y_true)
    for ii in xrange(0, n_Nor, 1):
        sub_z = T.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = T.concatenate([z, T.stack(T.sum(sub_z))])

    z = z[Num_d:]  # We need this step since we have used T.ones_like
    z = T.mean(z, axis=-1) +  0.00008*T.sum(sub_sum_l1) + 0.00008*T.sum(sub_l2)  # Final Loss f

    return z


adagrad=Adagrad(lr=0.01, epsilon=1e-08)

model.compile(loss=custom_objective, optimizer=adagrad)

print("Starting training...")

AllClassPath='/content/drive/MyDrive/DL_project/embedding_data'
# AllClassPath contains C3D features (.txt file)  of each video. Each text file contains 32 features, each of 4096 dimension
output_dir='/content/output'
# Output_dir is the directory where you want to save trained weights
weights_path = output_dir + 'weights.mat'
# weights.mat are the model weights that you will get after (or during) that training
model_path = output_dir + 'model.json'

if not os.path.exists(output_dir):
       os.makedirs(output_dir)

All_class_files= listdir(AllClassPath)
All_class_files.sort()
loss_graph =[]
num_iters = 20000
total_iterations = 0
batchsize=60
time_before = datetime.now()

for it_num in range(num_iters):

    AbnormalPath = os.path.join(AllClassPath, All_class_files[0])  # Path of abnormal already computed C3D features
    NormalPath = os.path.join(AllClassPath, All_class_files[1])    # Path of Normal already computed C3D features
    inputs, targets=load_dataset_Train_batch(AbnormalPath, NormalPath)  # Load normal and abnormal video C3D features
    batch_loss =model.train_on_batch(inputs, targets)
    loss_graph = np.hstack((loss_graph, batch_loss))
    total_iterations += 1
    if total_iterations % 20 == 1:
        #print "These iteration=" + str(total_iterations) + ") took: " + str(datetime.now() - time_before) + ", with loss of " + str(batch_loss)
        iteration_path = output_dir + 'Iterations_graph_' + str(total_iterations) + '.mat'
        savemat(iteration_path, dict(loss_graph=loss_graph))
    if total_iterations % 1000 == 0:  # Save the model at every 1000th iterations.
       weights_path = output_dir + 'weightsAnomalyL1L2_' + str(total_iterations) + '.mat'
       save_model(model, model_path, weights_path)


save_model(model, model_path, weights_path)