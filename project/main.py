import cliff_wakling as cw
import cv2
import numpy as np
import random


V = np.zeros(shape=48)
Q = np.zeros(shape=(48, 4))
discount_factory = 0.9
states_num = 48
action_num = 4
start_state = 36
end_state = 47
policy = np.zeros(shape=states_num)


def getReward(next_state , r):
    if next_state == end_state:
        return 100
    return r


def value_iteration(P):
    V_old = np.ones(shape=states_num)

    while np.abs(V-V_old).max()!=0:
        V_old = V.copy()
        for s in range(states_num):
            q_values = np.zeros(action_num)
            for a in range(action_num):
                for prob, next_state, reward, _ in P[s][a]:
                    q_values[a] += prob * (getReward(next_state,reward) + discount_factory * V_old[next_state])
            V[s] = max(q_values)
            policy[s] = np.argmax(q_values)


if __name__ == '__main__':
    # Create an environment
    env = cw.CliffWalking(render_mode="human")
    observation, info = env.reset(seed=30)

    # Define the maximum number of iterations
    max_iter_number = 1000

    value_iteration(env.P)

    turns = ["Up", "Right", "Down", "Left"]

    for i in range(len(policy)):
        turn = int(policy[i])
        print("i = " + str(i) + " ---> policy : " + str(turns[turn]))

    for __ in range(max_iter_number):
        action = policy[observation]

        next_state, reward, done, truncated, info = env.step(action)
        observation = next_state

        if done or truncated:
            observation, info = env.reset()

    # Close the environment
    env.close()
