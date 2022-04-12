import torch
import torch.nn as nn
import torch.nn.functional as F

class MilRankingLoss(nn.Module):
    def __init__(self):
        super(MilRankingLoss, self).__init__()
        #pass
    #def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #    pass
    #def mil_ranking(y_true, y_pred):
    def forward(self, y_true, y_pred):
        'Custom Objective function'

        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        print("MIL_Ranking")
        print(y_true)
        #print(y_true.type)
        print(y_pred)
        #print(y_pred.type)

        n_seg = 32  # Because we have 32 segments per video.
        #nvid = 60
        nvid = 1
        #n_exp = nvid / 2
        n_exp = nvid // 2
        Num_d=32*nvid


        sub_max = torch.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
        sub_sum_labels = torch.ones_like(y_true) # It is used to sum the labels in order to distinguish between normal and abnormal videos.
        sub_sum_l1=torch.ones_like(y_true)  # For holding the concatenation of summation of scores in the bag.
        sub_l2 = torch.ones_like(y_true) # For holding the concatenation of L2 of score in the bag.

        for ii in range(0, nvid, 1):
            # For Labels
            mm = y_true[ii * n_seg:ii * n_seg + n_seg]
            
            print(torch.sum(mm))
            sub_sum_labels = torch.cat([sub_sum_labels, torch.sum(mm)])  # Just to keep track of abnormal and normal vidoes

            # For Features scores
            Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
            sub_max = torch.cat([sub_max, torch.stack(torch.max(Feat_Score))])         # Keep the maximum score of scores of all instances in a Bag (video)
            sub_sum_l1 = torch.cat([sub_sum_l1, torch.stack(torch.sum(Feat_Score))])   # Keep the sum of scores of all instances in a Bag (video)

            z1 = torch.ones_like(Feat_Score)
            z2 = torch.cat([z1, Feat_Score])
            z3 = torch.cat([Feat_Score, z1])
            z_22 = z2[31:]
            z_44 = z3[:33]
            z = z_22 - z_44
            z = z[1:32]
            z = torch.sum(torch.sqr(z))
            sub_l2 = torch.cat([sub_l2, torch.stack(z)])


        # sub_max[Num_d:] means include all elements after Num_d.
        # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
        # z=x[4:]
        #[  6.  12.   7.  18.   9.  14.]

        sub_score = sub_max[Num_d:]  # We need this step since we have used torch.ones_like
        F_labels = sub_sum_labels[Num_d:] # We need this step since we have used torch.ones_like
        #  F_labels contains integer 32 for normal video and 0 for abnormal videos. This because of labeling done at the end of "load_dataset_Train_batch"



        # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
        # z=x[:4]
        # [ 2 4 3 9]... This shows 0 to 3 elements

        sub_sum_l1 = sub_sum_l1[Num_d:] # We need this step since we have used torch.ones_like
        sub_sum_l1 = sub_sum_l1[:n_exp]
        sub_l2 = sub_l2[Num_d:]         # We need this step since we have used torch.ones_like
        sub_l2 = sub_l2[:n_exp]


        indx_nor = theano.tensor.eq(F_labels, 32).nonzero()[0]  # Index of normal videos: Since we labeled 1 for each of 32 segments of normal videos F_labels=32 for normal video
        indx_abn = theano.tensor.eq(F_labels, 0).nonzero()[0]

        n_Nor=n_exp

        Sub_Nor = sub_score[indx_nor] # Maximum Score for each of abnormal video
        Sub_Abn = sub_score[indx_abn] # Maximum Score for each of normal video

        z = torch.ones_like(y_true)
        for ii in xrange(0, n_Nor, 1):
            sub_z = torch.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
            z = torch.cat([z, torch.stack(torch.sum(sub_z))])

        z = z[Num_d:]  # We need this step since we have used torch.ones_like
        z = torch.mean(z, axis=-1) +  0.00008*torch.sum(sub_sum_l1) + 0.00008*torch.sum(sub_l2)  # Final Loss f

        return z