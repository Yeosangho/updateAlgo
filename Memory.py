import numpy as np
import tensorflow as tf
import math
from Config import *
import random
import time
import gc
from numpy import float32, uint32, float64
from random import randint
import sys
np.random.seed(1)
tf.set_random_seed(1)

def sign(x): return 1 if x >= 0 else -1
# Sampling should not execute when the tree is not full !!!
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity, permanent_data=0):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # stores not probabilities but priorities !!!
        self.reverse_tree = np.zeros(2*capacity - 1)

        self.timetree = np.zeros(2*capacity - 1)
        self.demotree = np.zeros(2*capacity - 1)
        self.numtree = np.zeros(2*capacity - 1)
        #self.demotree = np.zeros(2 * capacity - 1)

        #self.numtree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # stores transitions

        self.data0 = np.zeros((capacity,84,84,3))
        self.data1 = [0] * capacity
        self.data2 = [0.0] * capacity
        self.data3 = np.zeros((capacity, 84, 84,3))
        self.data4 =  [False] * capacity
        self.data5 = [0.0] * capacity
        self.data6 = [0.0] * capacity
        self.data7 = np.zeros((capacity, 84, 84,3))
        self.data8 = [False] * capacity
        self.data9 = [0] * capacity
        self.data10 = [0] * capacity

        self.permanent_data = permanent_data  # numbers of data which never be replaced, for demo data protection
        assert 0 <= self.permanent_data <= self.capacity  # equal is also illegal
        self.full = False

        self.alpha = 0.7
        self.min_alpha = 0.1
        self.beta = 0.2
        self.min_beta = 0.0001
        self.gamma = 0.99
        self.alpha_decay_rate = 0.000002
        self.beta_decay_rate = 0.00001

        self.d_score_cahe = 0
        self.avg_val = 0
        self.avg_time = 0
        self.avg_demo = 0

    def __len__(self):
        return self.capacity if self.full else self.data_pointer


    def add(self, p,  time_stamp, is_demo, data, current_ts):
        demo = age = value = 0
        if not self.full:
            tree_idx = self.data_pointer + self.capacity - 1
            value = self.update(tree_idx, p, time_stamp, is_demo)
            #self.reverse_tree[0][1] = self.reverse_tree[0][1] + float64(1 - p).item()
            #print(self.reverse_tree[0][1])
            #print(float64(1-p).item())

            self.data0[self.data_pointer] = data[0]
            self.data1[self.data_pointer] = data[1]
            self.data2[self.data_pointer] = data[2]
            self.data3[self.data_pointer] = data[3]
            self.data4[self.data_pointer] = data[4]
            self.data5[self.data_pointer] = data[5]
            self.data6[self.data_pointer] = data[6]
            self.data7[self.data_pointer] = data[7]
            self.data8[self.data_pointer] = data[8]
            self.data9[self.data_pointer] = data[9]
            self.data10[self.data_pointer] = data[10]

            self.data_pointer += 1
            if self.data_pointer >= self.capacity:
                self.full = True
                self.data_pointer = self.data_pointer % self.capacity + self.permanent_data  # make sure demo data permanent
            #for i in range(data.__len__()):
            #    print(str(i)+" :" + str(type(data[i])))
            #print(data[0].shape)
            #print(data[3].shape)
            #print(data[7].shape)

        elif (self.full):

            experience_val = (1 - self.alpha) * self.total_reverse_p  + self.alpha * (self.capacity - (self.total_ts/current_ts)) - self.beta * self.total_d
            # print(self.numtree[1])
            v = np.random.uniform(0, experience_val)
            # print(v)

            tree_idx = self.chooseDeletedExperience(v, current_ts)
            #print(self.timetree[1])
            #print(del_tree_idx)
            self.data_pointer = tree_idx - self.capacity + 1
            age = self.data10[self.data_pointer]
            demo = int(self.data5[self.data_pointer])
            #self.data[self.data_pointer] = data  #####Memory Leak!!!

            self.data0[self.data_pointer] = data[0]
            self.data1[self.data_pointer] = data[1]
            self.data2[self.data_pointer] = data[2]
            self.data3[self.data_pointer] = data[3]
            self.data4[self.data_pointer] = data[4]
            self.data5[self.data_pointer] = data[5]
            self.data6[self.data_pointer] = data[6]
            self.data7[self.data_pointer] = data[7]
            self.data8[self.data_pointer] = data[8]
            self.data9[self.data_pointer] = data[9]
            self.data10[self.data_pointer] = data[10]


            value = self.update(tree_idx, p, time_stamp, is_demo)





            self.avg_val = self.avg_val + value
            self.avg_time = self.avg_time + age
            self.avg_demo = self.avg_demo + demo

    def update(self, tree_idx, p, ts=None, d=None):
        deleted_p = self.tree[tree_idx]
        change_p = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        deleted_reverse_p = self.reverse_tree[tree_idx]
        reverse_p = 1-p
        change_reverse_p = reverse_p - self.reverse_tree[tree_idx]
        #print(type(change_reverse_p))
        self.reverse_tree[tree_idx] = reverse_p

        change_num = 1 - self.numtree[tree_idx]
        self.numtree[tree_idx] = 1

        if ts is not None:
            change_ts = ts - self.timetree[tree_idx]
            change_d = d - self.demotree[tree_idx]
            self.timetree[tree_idx] = ts
            self.demotree[tree_idx] = d

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change_p
            self.reverse_tree[tree_idx] += change_reverse_p
            if(change_num == 1):
                self.numtree[tree_idx] += change_num
            if ts is not None:
                #print(change_ts)
                self.timetree[tree_idx] = self.timetree[tree_idx] + change_ts
                #print(type(self.reverse_tree[1][tree_idx]))
                self.demotree[tree_idx] += change_d
                #print(type(change_d))



        return deleted_p

    def chooseDeletedExperience(self, v, current_ts):
        parent_idx = 0
        #start_time = time.time()
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1
        #ts = self.timetree[left_child_idx]
        demo = self.demotree[left_child_idx]
        num = self.numtree[left_child_idx]

        while( left_child_idx < len(self.tree)):
            value = (1 - self.alpha) * self.reverse_tree[left_child_idx] + self.alpha * (self.numtree[left_child_idx] - (self.timetree[
                left_child_idx]/current_ts)) + self.beta * self.demotree[left_child_idx]
            if v <=   value:
                parent_idx = left_child_idx
            else:
                v -=  value
                parent_idx = right_child_idx
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
        return parent_idx


    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1


        return leaf_idx, self.tree[leaf_idx], [self.data0[data_idx],
                                              self.data1[data_idx],
                                              self.data2[data_idx],
                                              self.data3[data_idx],
                                              self.data4[data_idx],
                                              self.data5[data_idx],     
                                              self.data6[data_idx],
                                              self.data7[data_idx],
                                              self.data8[data_idx],
                                              self.data9[data_idx],
                                              self.data10[data_idx]]

    def anneal_alpha_beta(self, delta, actor_num, sub_train_iter):
        #print(sign(delta))
        delta = sign(delta)*math.log(abs(delta)+1)
        #print(delta)
        epsilon = 0.001
        for i in range(sub_train_iter):
            self.d_score_cahe = self.gamma * self.d_score_cahe + (1-self.gamma) * (delta*delta)
            if(self.alpha > self.min_alpha):
                alpha_ada_decay_rate = (self.alpha_decay_rate * delta) / ((self.d_score_cahe**(1/2))+epsilon)
                self.alpha = (self.alpha - (1/actor_num)* alpha_ada_decay_rate * self.alpha)
                if(self.alpha < self.min_alpha) :
                    self.alpha = self.min_alpha
            if(self.beta > self.min_beta):
                beta_ada_decay_rate = (self.beta_decay_rate * delta) / (self.d_score_cahe**(1/2)+epsilon)
                self.beta = (self.beta - (1/actor_num)* beta_ada_decay_rate * self.beta)
                if(self.beta < self.min_beta) :
                    self.beta = self.min_beta
        print(self.alpha)
        print(self.beta)
    @property
    def total_p(self):
        return self.tree[0]
    @property
    def total_ts(self):
        return self.timetree[0]
    @property
    def total_d(self):
        return self.demotree[0]
    @property
    def total_reverse_p(self):
        return self.reverse_tree[0]

class Memory(object):

    epsilon = 0.001  # small amount to avoid zero priority
    demo_epsilon = 1.0  # 1.0  # extra
    alpha = 0.4  # [0~1] convert the importance of TD error to priority
    beta = 0.6  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error


    def __init__(self, capacity, permanent_data=0):

        self.permanent_data = permanent_data
        self.tree = SumTree(capacity, permanent_data)

    def __len__(self):
        return len(self.tree)

    def full(self):
        return self.tree.full

    def store(self, transition, abs_errors, is_demo, time_stamp, current_ts):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        is_demo = 1 - is_demo
        current_ts = current_ts + 1
        #log_current_ts = np.log(current_ts + 1)
        #print(ps)
        self.tree.add(ps[0], time_stamp, is_demo, transition, current_ts)  # set the max_p for new transition
        #value, age, demo = self.tree.add(ps[0], log_time_stamp, is_demo, transition, log_current_ts)  # set the max_p for new transition
    def sample(self, n, current_ts):
        assert self.full()
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, 11), dtype=object)
        ISWeights = np.empty((n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        if(min_prob <= 0):
            min_prob = self.epsilon
        pmore = 0
        p2more = 0


        for i in range(n):
            v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
            #v = np.random.uniform(0, self.tree.total_p )
            #v = 0.001
            #v = self.tree.total_p - 0.001
            idx, p, data = self.tree.get_leaf(v)  # note: idx is the index in self.tree.tree
            #print("p :" + str(p))
            #idx2, p2, _ = self.tree.chooseDeletedExperience(v)
            #print("p2 :" + str(p2))
            #if(p > p2):
            #    pmore = pmore +1
            #else :
            #    p2more = p2more +1
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        #print(str(pmore) + ":" + str(p2more))
        return b_idx, b_memory, ISWeights  # note: b_idx stores indexes in self.tree.tree, not in self.tree.data !!!

    # update priority
    def batch_update(self, tree_idxes, abs_errors, time_stamp, isdemo):
        # priorities of demo transitions are given a bonus of demo_epsilon, to boost the frequency that they are sampled
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        #time_stamp = np.asarray(time_stamp)
        #time_stamp = np.log(time_stamp + 1)
        isdemo = np.asarray(isdemo)
        #log_time_stamp = np.log(time_stamp +1)
        #print(time_stamp.shape)
        #print(type())
        for ti, p, d in zip(tree_idxes, ps, isdemo):
            #self.tree.update(ti, p,  time_stamp, d)
            self.tree.update(ti, p)

    def update_alpha_and_beta(self, dscore, actor_num, sub_train_iter):
        self.tree.anneal_alpha_beta(dscore, actor_num, sub_train_iter)

