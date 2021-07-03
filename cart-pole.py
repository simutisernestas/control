import scipy.signal
import numpy as np
import gym

m = 0.1  # pole mass
M = 1.0  # cart mass
L = 0.5  # cart length
d = 0.0  # zero damping
g = 9.81

# dx = Ax + Bu

# linearized dynamics around straight up cart
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, d/(M*L), (m+M)*g/(M*L), 0],
])
# control
B = np.array([
    [0],
    [1/M],
    [0],
    [1/(M*L)],
])

eigs = np.array([-1.0, -.5, -.2, -1.3])
k = scipy.signal.place_poles(A, B, eigs)
# print(k.gain_matrix, k.computed_poles)

env = gym.make('CartPole-v0')
x = env.reset()
for t in range(200):
    env.render()

    # all zeros illustrate desired position, has no effect here
    # however we wish to drive the system to that state
    u = k.gain_matrix.dot(
        (np.array(x).T - np.array([0, 0, 0, 0]).T))

    if u < 0:
        action = 0
    else:
        action = 1

    x, _, done, _ = env.step(action)

    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
