import numpy as np
import tensorflow as tf
import math 
np.random.seed(1)
tf.set_random_seed(1)

def sign(x): return 1 if x >= 0 else -1
# Sampling should not execute when the tree is not full !!!
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity, permanent_data=0):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # stores not probabilities but priorities !!!
        self.timetree = np.zeros(2 * capacity - 1)
        self.demotree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # stores transitions
        self.permanent_data = permanent_data  # numbers of data which never be replaced, for demo data protection
        assert 0 <= self.permanent_data <= self.capacity  # equal is also illegal
        self.full = False

        self.alpha = 0.6
        self.min_alpha = 0.01
        self.beta = 0.2
        self.min_beta = 0.001
        self.gamma = 0.99
        self.alpha_decay_rate = 0.00001
        self.beta_decay_rate = 0.00001

        self.d_score_cahe = 0

    def __len__(self):
        return self.capacity if self.full else self.data_pointer

    def add(self, p,  time_stamp, is_demo, data, current_ts):
        demo = age = value = 0
        if not self.full:
            tree_idx = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = data
            value = self.update(tree_idx, p, time_stamp, is_demo)
            self.data_pointer += 1
            if self.data_pointer >= self.capacity:
                self.full = True
                self.data_pointer = self.data_pointer % self.capacity + self.permanent_data  # make sure demo data permanent
        elif (self.full):
            if(current_ts == 0):
                experience_val = (1-self.alpha)*self.total_p  + self.beta *is_demo
            else :
                experience_val = (1-self.alpha)*self.total_p + (self.alpha)*(self.total_ts/current_ts) + self.beta *is_demo
            #v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
            del_tree_idx  = self.chooseDeletedExperience(experience_val, current_ts)
            del_data_idx = del_tree_idx - self.capacity + 1
            deleteddata = self.data[del_tree_idx - self.capacity +1]
            self.data[del_data_idx]  = data
            value = self.update(del_tree_idx, p, time_stamp, is_demo)
            demo = int(deleteddata[5])    
            age = deleteddata[10]                 
        return value, age, demo
    def update(self, tree_idx, p, ts, d):
        deleted_p = self.tree[tree_idx]
        change_p = p - self.tree[tree_idx]

        deleted_ts = self.timetree[tree_idx]
        change_ts = ts - self.timetree[tree_idx]


        deleted_d = self.demotree[tree_idx]
        change_d = d - self.demotree[tree_idx]

        self.tree[tree_idx] = p
        self.timetree[tree_idx] = ts
        self.demotree[tree_idx] = d
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change_p
            self.timetree[tree_idx] += change_ts
            self.demotree[tree_idx] += change_d

        return deleted_p
    def chooseDeletedExperience(self, v, current_ts):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.calc_exp_val(self.tree[left_child_idx], self.timetree[left_child_idx], self.demotree[left_child_idx], current_ts):
                parent_idx = right_child_idx
            else:
                v -= self.calc_exp_val(self.tree[left_child_idx], self.timetree[left_child_idx], self.demotree[left_child_idx], current_ts)
                parent_idx = left_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx


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
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    def calc_exp_val(self, p, ts, d, current_ts):
        if(current_ts == 0):
            exp_val = (1-self.alpha)*p + self.beta *d
        else :
            exp_val = (1-self.alpha)*p + (self.alpha)*(ts/current_ts) + self.beta *d
        return exp_val

    def anneal_alpha_beta(self, delta, actor_num, sub_train_iter):
        delta = sign(delta)*math.log(abs(delta)+1)
        epsilon = 0.001
        self.d_score_cahe = self.gamma * self.d_score_cahe + (1-self.gamma) * (delta*delta)
        if(self.alpha > self.min_alpha):
            alpha_ada_decay_rate = (self.alpha_decay_rate * delta) / (self.d_score_cahe**(1/2)+epsilon)
            self.alpha = (self.alpha - (sub_train_iter/actor_num)* alpha_ada_decay_rate)
            if(self.alpha < self.min_alpha) :
                self.alpha = self.min_alpha
        if(self.beta > self.min_beta):
            beta_ada_decay_rate = (self.beta_decay_rate * delta) / (self.d_score_cahe**(1/2)+epsilon)
            self.beta = (self.beta - (sub_train_iter/actor_num)* beta_ada_decay_rate)
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

class Memory(object):

    epsilon = 0.000001  # small amount to avoid zero priority
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
        log_time_stamp = np.log(time_stamp+1)
        log_current_ts = np.log(current_ts + 1)
        #print(ps)
        value, age, demo = self.tree.add(ps[0], log_time_stamp, is_demo, transition, log_current_ts)  # set the max_p for new transition
        return value, age, demo
    def sample(self, n):
        assert self.full()
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, self.tree.data[0].size), dtype=object)
        ISWeights = np.empty((n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        assert min_prob > 0
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
        time_stamp = np.asarray(time_stamp)
        time_stamp = np.log(time_stamp + 1)
        isdemo = np.asarray(isdemo)
        #log_time_stamp = np.log(time_stamp +1)
        #print(time_stamp.shape)
        #print(type())
        for ti, p, d in zip(tree_idxes, ps, isdemo):
            self.tree.update(ti, p,  time_stamp, d)

    def update_alpha_and_beta(self, dscore, actor_num, sub_train_iter):
        self.tree.anneal_alpha_beta(dscore, actor_num, sub_train_iter)
