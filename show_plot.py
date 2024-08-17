import numpy as np
import matplotlib.pyplot as plt #在线显示图像
import rl_utils
from matplotlib import pyplot
if __name__ == "__main__":

    dqn_data =[]
    dqn_data1 = np.load(f"./data/alpha05action11/q_value/DQN_Pendulum-v0_0.npy")
    dqn_data.append(rl_utils.moving_average(dqn_data1, 5))
    #'''
    dqn_data1 = np.load(f"./data/alpha05action11/q_value/DQN_Pendulum-v0_1.npy")
    dqn_data.append(rl_utils.moving_average(dqn_data1, 5))

    dqn_data1 = np.load(f"./data/alpha05action11/q_value/DQN_Pendulum-v0_2.npy")
    dqn_data.append(rl_utils.moving_average(dqn_data1, 5))

    dqn_data1 = np.load(f"./data/alpha05action11/q_value/DQN_Pendulum-v0_3.npy")
    dqn_data.append(rl_utils.moving_average(dqn_data1, 5))

    dqn_data1 = np.load(f"./data/alpha05action11/q_value/DQN_Pendulum-v0_4.npy")
    dqn_data.append(rl_utils.moving_average(dqn_data1, 5))

    dqn_data1 = np.load(f"./data/alpha05action11/q_value/DQN_Pendulum-v0_5.npy")
    dqn_data.append(rl_utils.moving_average(dqn_data1, 5))
    #'''

    ddqn_data = []
    ddqn_data1 = np.load(f"./data/alpha05action11/q_value/DoubleDQN_Pendulum-v0_0.npy")
    ddqn_data.append(rl_utils.moving_average(ddqn_data1, 5))
    #'''
    ddqn_data1 = np.load(f"./data/alpha05action11/q_value/DoubleDQN_Pendulum-v0_1.npy")
    ddqn_data.append(rl_utils.moving_average(ddqn_data1, 5))
    ddqn_data1 = np.load(f"./data/alpha05action11/q_value/DoubleDQN_Pendulum-v0_2.npy")
    ddqn_data.append(rl_utils.moving_average(ddqn_data1, 5))
    ddqn_data1 = np.load(f"./data/alpha05action11/q_value/DoubleDQN_Pendulum-v0_3.npy")
    ddqn_data.append(rl_utils.moving_average(ddqn_data1, 5))
    ddqn_data1 = np.load(f"./data/alpha05action11/q_value/DoubleDQN_Pendulum-v0_4.npy")
    ddqn_data.append(rl_utils.moving_average(ddqn_data1, 5))
    ddqn_data1 = np.load(f"./data/alpha05action11/q_value/DoubleDQN_Pendulum-v0_5.npy")
    ddqn_data.append(rl_utils.moving_average(ddqn_data1, 5))

    #'''

    udqn_data =[]
    udqn_data1 = np.load(f"./data/alpha05action11/q_value/u1DQN_Pendulum-v0_0.npy")
    udqn_data.append(rl_utils.moving_average(udqn_data1, 5))
    #'''
    udqn_data1 = np.load(f"./data/alpha05action11/q_value/u1DQN_Pendulum-v0_1.npy")
    udqn_data.append(rl_utils.moving_average(udqn_data1, 5))

    udqn_data1 = np.load(f"./data/alpha05action11/q_value/u1DQN_Pendulum-v0_2.npy")
    udqn_data.append(rl_utils.moving_average(udqn_data1, 5))

    udqn_data1 = np.load(f"./data/alpha05action11/q_value/u1DQN_Pendulum-v0_3.npy")
    udqn_data.append(rl_utils.moving_average(udqn_data1, 5))

    udqn_data1 = np.load(f"./data/alpha05action11/q_value/u1DQN_Pendulum-v0_4.npy")
    udqn_data.append(rl_utils.moving_average(udqn_data1, 5))

    udqn_data1 = np.load(f"./data/alpha05action11/q_value/u1DQN_Pendulum-v0_5.npy")
    udqn_data.append(rl_utils.moving_average(udqn_data1, 5))
    #'''

    dqn_avg = np.mean(dqn_data, axis=0)
    dqn_std = np.std(dqn_data, axis=0)
    dqn_r1 = list(map(lambda x: x[0] - x[1], zip(dqn_avg, dqn_std)))  # 上方差
    dqn_r2 = list(map(lambda x: x[0] + x[1], zip(dqn_avg, dqn_std)))  # 下方差

    ddqn_avg = np.mean(ddqn_data, axis=0)
    ddqn_std = np.std(ddqn_data, axis=0)
    ddqn_r1 = list(map(lambda x: x[0] - x[1], zip(ddqn_avg, ddqn_std)))  # 上方差
    ddqn_r2 = list(map(lambda x: x[0] + x[1], zip(ddqn_avg, ddqn_std)))  # 下方差

    udqn_avg = np.mean(udqn_data, axis=0)
    udqn_std = np.std(udqn_data, axis=0)
    udqn_r1 = list(map(lambda x: x[0] - x[1], zip(udqn_avg, udqn_std)))  # 上方差
    udqn_r2 = list(map(lambda x: x[0] + x[1], zip(udqn_avg, udqn_std)))  # 下方差




    palette = pyplot.get_cmap('Set1')
    frame = []
    frame = list(range(len(dqn_avg)))
    plt.plot(frame, dqn_avg, color=palette(1), label='DQN')
    plt.fill_between(frame,dqn_r1, dqn_r2, color=palette(1),  alpha=0.2)

    plt.plot(frame, ddqn_avg, color=palette(2), label='DDQN')
    plt.fill_between(frame,ddqn_r1, ddqn_r2, color=palette(2), alpha=0.2)
    #'''
    plt.plot(frame, udqn_avg, color=palette(3), label='U1DQN')
    plt.fill_between(frame, udqn_r1, udqn_r2, color=palette(3), alpha=0.2)
    #'''
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frame')
    plt.ylabel('Q value')
    plt.title('DQN vs DoubleDQN vs U1DQN on Pendulum')
    plt.legend(loc='upper left')
    plt.show()


