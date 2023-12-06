import cliff_wakling as cw
import cv2
import numpy as np
import random


V = np.zeros(shape=48)
Q = np.zeros(shape=(48, 4))
discount_factory = 0.9


def SetQTable(state, next_state, prob, action, reward):
    pass


if __name__ == '__main__':
    # Create an environment
    env = cw.CliffWalking(render_mode="human")
    observation, info = env.reset(seed=30)

    # Define the maximum number of iterations
    max_iter_number = 1000

    for key , value in env.P.items():
        print("key : "+str(key)+" --- value : "+str(value))

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
