import numpy as np
import random
import operator
import matplotlib.pyplot as plt

def state_value_function(p,r,v):
    gamma=0.9
    while True:
        value=r.T+gamma*p*v
        if(value==v).all():
            break
        v=value
    return v

def random_pick(some_list):          # randomly pick to go to different state
    probabilities=[0.5,0.5]
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
         cumulative_probability += item_probability
         if x < cumulative_probability:
               break
    return item

def pick_state(index):
    route_list=((trans_matrix[index,:])).tolist()
    next_list=[]
    for i in range(7):
        if route_list[0][i]!=0:
            next_list.append(i)
    return state_list[random_pick(next_list)]

def td():
    Vs = [0, 0, 0, 0, 0, 0, 0]
    i = 0
    while True:
        i=i+1
        state = state_list[3]
        subtuple = [3]
        old_Vs = Vs.copy()
        while True:
            if state != 0 and state != 6:
                #sample tuple
                state = pick_state(state_list.index(state))
                subtuple.append(state)
                subtuple.append(reward_matrix[subtuple[0], subtuple[1]])
                #update V(s)
                Vs[subtuple[0]] = Vs[subtuple[0]] + alpha * (subtuple[2] + (gamma * Vs[subtuple[1]]) - Vs[subtuple[0]])
                subtuple = []
                subtuple.append(state)
            else:
                break
        print(i, " ", Vs[1:6])  # print iterations and V(s)
        # determine if V(s) no change
        if operator.eq(old_Vs, Vs):
            if operator.eq(Vs, [0, 0, 0, 0, 0, 0, 0]):
                continue
            else:
                break


#t1 a b c d e t2
trans_matrix=np.mat([[0,0,0,0,0,0,0],
                     [0.5,0,0.5,0,0,0,0],
                     [0,0.5,0,0.5,0,0,0],
                     [0,0,0.5,0,0.5,0,0],
                     [0,0,0,0.5,0,0.5,0],
                     [0,0,0,0,0.5,0,0.5],
                     [0,0,0,0,0,0,0]])
gamma=0.9
alpha=0.01
alpha_list=[0.01,0.03,0.05]
state_list=[0,1,2,3,4,5,6]
reward_matrix=np.mat([[-1,-1,-1,-1,-1,-1,-1],
                      [0,-1,0,-1,-1,-1,-1],
                      [-1,0,-1,0,-1,-1,-1],
                      [-1,-1,0,-1,0,-1,-1],
                      [-1,-1,-1,0,-1,0,-1],
                      [-1,-1,-1,-1,0,-1,1],
                      [-1,-1,-1,-1,-1,-1,-1]])
R_matrix=np.mat([0,0,0,0,0,0,1])
v_function=np.mat(np.zeros((7,1)))
true_value=(state_value_function(trans_matrix,R_matrix,v_function))[1:6]
td()
