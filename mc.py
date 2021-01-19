import numpy as np
import random
import operator
import matplotlib.pyplot as plt

def state_value_function(p,r,v):
    while True:
        value=r.T+gamma*p*v
        if(value==v).all():
            break
        v=value
    return v

# randomly select index of next state
def random_pick(some_list):
    probabilities=[0.5,0.5]#state transition matrix
    x = random.uniform(0,1)#create a random figure
    probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
         probability += item_probability
         if x < probability:
               break
    return item

#randomly pick next state
def pick_state(index):
    route_list=((trans_matrix[index,:])).tolist()
    next_list=[]
    for i in range(7):
        if route_list[0][i]!=0:
            next_list.append(i)#get the next state that may reach
    return state_list[random_pick(next_list)]

def compute_return(s,e_list):
    #record Gt of every state in one episode at each time
    g_list=[]
    g=0
    for i in range(len(e_list)-1):
        #computer Gt
        g=g*gamma+reward_matrix[e_list[len(e_list)-2-i],e_list[len(e_list)-1-i]]
        g_list.append(g)
    #revrese the list
    g_list.reverse()
    # print('episode:')
    # print(e_list)
    # print('Gt of one episode:')
    # print(g_list)
    gs_list=[]
    #get all Gt of one state in one episode
    for i in range(len(g_list)):
        if e_list[i]==s:
            gs_list.append(g_list[i])
    #print(gs_list)
    return gs_list

def mc():
    Ns = [0, 0, 0, 0, 0]
    Vs = [0, 0, 0, 0, 0]
    i=0
    while True:
        i=i+1
        #sample episode
        state = state_list[3]
        episode = [3]
        while True:
            if state != 0 and state != 6:
                state = pick_state(state_list.index(state))
                episode.append(state)
            else:
                break
        old_Vs = Vs.copy()
        #uodate V(s)
        for x in state_list:
            if x == 0 or x == 6:
                continue
            else:
                #obtain return
                gs = compute_return(x, episode)
                if gs:
                    for reward in gs:
                        Ns[x - 1] = Ns[x - 1] + 1
                        #compute V(s)
                        Vs[x - 1] = Vs[x - 1] + (reward - Vs[x - 1]) * alpha
                else:
                    continue
        print(i," ",Vs)#print iterations and V(s)
        # determine if V(s) no change
        if operator.eq(old_Vs,Vs):
            if operator.eq(Vs,[0,0,0,0,0]):
                continue
            else:
                break

def mc_2():
    Ns = [0, 0, 0, 0, 0]
    Vs = [0, 0, 0, 0, 0]
    i=0
    while True:
        i=i+1
        #sample episode
        state = state_list[3]
        episode = [3]
        while True:
            if state != 0 and state != 6:
                state = pick_state(state_list.index(state))
                episode.append(state)
            else:
                break
        old_Vs = Vs.copy()
        #uodate V(s)
        for x in state_list:
            if x == 0 or x == 6:
                continue
            else:
                #obtain return
                gs = compute_return(x, episode)
                if gs:
                    for reward in gs:
                        Ns[x - 1] = Ns[x - 1] + 1
                        #compute V(s)
                        Vs[x - 1] = Vs[x - 1] + (reward - Vs[x - 1]) * alpha
                else:
                    continue
        print(i," ",Vs)#print iterations and V(s)
        # determine if V(s) no change
        if (np.around(old_Vs,3)==np.around(Vs,3)).all():
            if operator.eq(Vs,[0,0,0,0,0]):
                continue
            else:
                break


#order: t1 a b c d e t2
trans_matrix=np.mat([[0,0,0,0,0,0,0],
                     [0.5,0,0.5,0,0,0,0],
                     [0,0.5,0,0.5,0,0,0],
                     [0,0,0.5,0,0.5,0,0],
                     [0,0,0,0.5,0,0.5,0],
                     [0,0,0,0,0.5,0,0.5],
                     [0,0,0,0,0,0,0]])
#print(trans_matrix)
gamma=0.9
rate=0.01
alpha=0.05
#list to represent each state(t1 a b c d e t2)
state_list=[0,1,2,3,4,5,6]
reward_matrix=np.mat([[-1,-1,-1,-1,-1,-1,-1],
                      [0,-1,0,-1,-1,-1,-1],
                      [-1,0,-1,0,-1,-1,-1],
                      [-1,-1,0,-1,0,-1,-1],
                      [-1,-1,-1,0,-1,0,-1],
                      [-1,-1,-1,-1,0,-1,1],
                      [-1,-1,-1,-1,-1,-1,-1]])
R_matrix=np.mat([0,0,0,0,0,0,1])
a=[1,1,1]
b=[1,1,1]
mc_2()
