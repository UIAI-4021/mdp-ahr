import cliff_wakling as cw
import numpy as np

V = np.zeros(shape=48)
discount_factory = 0.999
states_num = 48
action_num = 4
start_state = 36
end_state = 47
policy = np.zeros(shape=states_num)


def getReward(next_state , r):
    if next_state == end_state:
        return 4000
    return r


def check_cliff(state):
    positions = env.cliff_positions
    row = state // 12
    column = state % 12
    value = (row, column)

    if value in positions:
        return True
    return False


def setCliffs(env):
    positions = env.cliff_positions

    for cliffs in positions:
        index = cliffs[0] * 12 + cliffs[1]
        V[index] = -100


def optimal_policy(env):
    V_old = np.ones(shape=states_num)
    P = env.P
    V[end_state] = 4000
    setCliffs(env)

    while np.abs(V - V_old).max() != 0:
    # for _ in range(1000):
        V_old = V.copy()
        for s in range(states_num):
            if not check_cliff(s) and s != end_state:
                q_values = np.zeros(shape=action_num)
                for q in range(action_num):
                    for a in range(action_num):
                        if abs(q - a) != 2:
                            P_value = P[s][a][0]
                            next_state = P_value[1]
                            reward = P_value[2]

                            q_values[q] += (1/3) * (getReward(next_state , reward) + discount_factory * V_old[next_state])

                V[s] = max(q_values)
                policy[s] = np.argmax(q_values)


if __name__ == '__main__':
    # Create an environment
    env = cw.CliffWalking(render_mode="human")
    observation, info = env.reset(seed=30)

    # Define the maximum number of iterations
    max_iter_number = 1000

    optimal_policy(env)

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
            print("GET GOAL")

    # Close the environment
    env.close()
