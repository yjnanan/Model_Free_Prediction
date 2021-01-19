import numpy as np
import random
import matplotlib.pyplot as plt
# define the transmission matrix
# States : T1,A,B,C,D,E,T2
def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
         cumulative_probability += item_probability
         if x < cumulative_probability:
               break
    return item

trans_matrix = np.mat ([[0, 0, 0, 0, 0, 0, 0],
                      [0.5, 0, 0.5, 0, 0, 0, 0],
                      [0, 0.5, 0, 0.5, 0, 0, 0],
                      [0, 0, 0.5, 0, 0.5, 0, 0],
                      [0, 0, 0, 0.5, 0, 0.5, 0],
                      [0, 0, 0, 0, 0.5, 0, 0.5],
                      [0, 0, 0, 0, 0, 0, 0]])

# enumerate all the states
states = [0, 1, 2, 3, 4, 5, 6]

# reward
reward = [0, 0, 0, 0, 0, 0, 1]
r = np.zeros((7, 7))
r[5, 6] = 1

# discount factor
gamma = 1

# start_state = states[3]
# index = 3
# current_state = start_state
alpha = 0.01

value = [0, 0, 0, 0, 0, 0, 0]
old_value = [0, 0, 0, 0, 0, 0, 0]
for i in range(50000):
    start_state = states[3]
    index = 3
    current_state = start_state

    while True:
        single_tuple = [current_state]
        step = random_pick([-1, 1], [0.5, 0.5])
        index = index + step
        next_state = states[index]
        single_tuple.append(next_state)
        old_value = value.copy()

        # print(single_tuple)

        TD_target = r[current_state, next_state] + gamma * value[single_tuple[1]]
        # TD_target = reward[single_tuple[1]] + gamma * value[single_tuple[1]]
        value[single_tuple[0]] = value[single_tuple[0]] + alpha * (TD_target - value[single_tuple[0]])
        current_state = next_state
        if current_state == 0 or current_state == 6:
            break

    # if sum(old_value) > 1.5:
    #     if (np.round(old_value, 3) == np.round(value, 3)).all():
    #         break
print(old_value)
print(value)

# x = [1,2,3,4,5]
# y = [1/6,2/6,3/6,4/6,5/6]
# plt.plot(x,value[1:6],'ro-')
# plt.plot(x,y,'bo-')
# plt.show()