import numpy as np

def state_value_function(p,r,v):
    gamma=0.9
    while True:
        value=r.T+gamma*p*v
        if(value==v).all():
            break
        v=value
    return v

if __name__ =='__main__':
    #c1 c2 c3 pass pub fb sleep
    #probability matrix
    P_matrix=np.mat([[0,0,0,0,0,0,0],
                     [0.5,0,0.5,0,0,0,0],
                     [0,0.5,0,0.5,0,0,0],
                     [0,0,0.5,0,0.5,0,0],
                     [0,0,0,0.5,0,0.5,0],
                     [0,0,0,0,0.5,0,0.5],
                     [0,0,0,0,0,0,0]])
    R_matrix=np.mat([0,0,0,0,0,0,1])
    v_function=np.mat(np.zeros((7,1)))
    print(state_value_function(P_matrix,R_matrix,v_function))