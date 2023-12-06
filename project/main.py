import cliff_wakling as cw
import cv2
import numpy as np
import random


V = np.zeros(shape=48)
Q = np.zeros(shape=(48, 4))
discount_factory = 0.9
states_num = 48
action_num = 4
policy = np.zeros(shape=states_num)


def SetQTable(state, next_state, prob, action, reward):
    r = reward
    if state == next_state:
        r = -100
    Q[state][action] = prob * (r + discount_factory * V[next_state])


if __name__ == '__main__':
    # Create an environment
    env = cw.CliffWalking(render_mode="human")
    observation, info = env.reset(seed=30)

    # Define the maximum number of iterations
    max_iter_number = 1000

    for key , value in env.P.items():
        print("key : "+str(key)+" --- value : "+str(value))
    P_values = env.P

    for __ in range(500):
        V_old=V.copy()
        for s in range(states_num):
            q_values = np.zeros(action_num)
            for a in range(action_num):
                for prob, next_state, reward, _ in P_values[s][a]:
                    # print("prob : "+str(prob)+" --- next_state : "+str(next_state)+" --- reward : "+str(reward))
                    # input()
                    q_values[a] += prob * (reward + discount_factory * V_old[next_state])
            V[s] = max(q_values)
            policy[s] = np.argmax(q_values)

    for __ in range(max_iter_number):
        # TODO: Implement the agent policy here
        # Note: .sample() is used to sample random action from the environment's action space
        #
        # Choose an action (Replace this random action with your agent's policy)
        action = env.action_space.sample()

        # Perform the action and receive feedback from the environment
        next_state, reward, done, truncated, info = env.step(action)
        observation = next_state

        if done or truncated:
            observation, info = env.reset()

    # Close the environment
    env.close()
