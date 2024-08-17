import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    ''' DQN算法,包括Double DQN '''
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        elif self.dqn_type == 'DQN': # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标

        elif self.dqn_type == 'u2DQN':
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

            max_action = self.target_q_net(next_states).max(1)[1].view(-1, 1)

            max_next_q_values_1 = self.target_q_net(next_states).gather(1, max_action)
            max_next_q_values_2 = self.q_net(next_states).gather(1, max_action)
            u = abs(max_next_q_values_1 - max_next_q_values_2)
            q_targets = rewards + self.gamma * (max_next_q_values - 0.5 * u) * (1 - dones)


        elif self.dqn_type == 'u1DQN':
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            next_state_q_values = self.target_q_net(next_states)
            # 找到最大值及其索引
            max_action_idx = next_state_q_values.max(dim=1)[1]

            # 将最大值对应的位置设置为负无穷，排除这个最大值
            for j in range(batch_size):
                next_state_q_values[j, max_action_idx[j]] = float('-inf')

            # 找到排除最大值后的最大值，即为第二大的 Q 值
            scd_max_next_q_values = next_state_q_values.max(dim=1)[0].view(-1, 1)
            u = max_next_q_values - scd_max_next_q_values

            q_targets = rewards + self.gamma * (max_next_q_values - 0.5*u) * (1 - dones)  # TD误差目标
            for j in range(batch_size):
                next_state_q_values[j, max_action_idx[j]] = max_next_q_values[j]

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'Pendulum-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 25 # 将连续动作分成11个离散动作


def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size,seed):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env,
                                                   agent.action_dim)
                    next_state, reward, done, _ = env.step([action_continuous])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    file_name = f"{agent.dqn_type}_{env_name}_{seed}"
    np.save(f"./data/test/return/{file_name}", return_list)
    np.save(f"./data/test/q_value/{file_name}", max_q_value_list)
    return return_list, max_q_value_list

if __name__ == '__main__':
    dqn_return_list = []
    dqn_max_q_value_list = []
    ddqn_return_list = []
    ddqn_max_q_value_list = []
    dqn_mv_return = []
    ddqn_mv_return = []
    udqn_return_list = []
    udqn_mv_return = []
    udqn_max_q_value_list = []

    for i in range(6):
        # uDQN
        random.seed(i)
        np.random.seed(i)
        env.seed(i)
        torch.manual_seed(i)
        udqn_replay_buffer = rl_utils.ReplayBuffer(buffer_size)
        udqn_agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                         target_update, device, "u1DQN")
        udqn_return, udqn_q_value = train_DQN(udqn_agent, env, num_episodes, udqn_replay_buffer, minimal_size,
                                              batch_size, i)
        udqn_return_list.append(udqn_return)
        udqn_max_q_value_list.append(udqn_q_value)
        udqn_mv_return.append(rl_utils.moving_average(udqn_return, 5))
        # dqn

        random.seed(i)
        np.random.seed(i)
        env.seed(i)
        torch.manual_seed(i)
        dqn_replay_buffer = rl_utils.ReplayBuffer(buffer_size)
        dqn_agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device, "DQN")
        dqn_return, dqn_q_value = train_DQN(dqn_agent, env, num_episodes, dqn_replay_buffer, minimal_size, batch_size, i)
        dqn_return_list.append(dqn_return)
        dqn_max_q_value_list.append(dqn_q_value)
        dqn_mv_return.append(rl_utils.moving_average(dqn_return, 5))

        # ddqn
        random.seed(i)
        np.random.seed(i)
        env.seed(i)
        torch.manual_seed(i)
        ddqn_replay_buffer = rl_utils.ReplayBuffer(buffer_size)
        ddqn_agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device, "DoubleDQN")
        ddqn_return, ddqn_q_value = train_DQN(ddqn_agent, env, num_episodes, ddqn_replay_buffer, minimal_size, batch_size, i)
        ddqn_return_list.append(ddqn_return)
        ddqn_max_q_value_list.append(ddqn_q_value)
        ddqn_mv_return.append(rl_utils.moving_average(ddqn_return, 5))




    dqn_episodes_list = list(range(len(dqn_return_list[0])))
    ddqn_episodes_list = list(range(len(ddqn_return_list[0])))
    
    mean_dqn_mv_return = np.mean(dqn_mv_return, axis=0)
    mean_ddqn_mv_return = np.mean(ddqn_mv_return, axis=0)



    udqn_episode_list = list(range(len(udqn_return_list[0])))
    mean_udqn_mv_return = np.mean(udqn_mv_return, axis=0)


    plt.plot(dqn_episodes_list, mean_dqn_mv_return, label='DQN')
    plt.plot(ddqn_episodes_list, mean_ddqn_mv_return, label='doubleDQN')


    plt.plot(udqn_episode_list, mean_udqn_mv_return, label='uDQN')
    plt.xlabel('Episodes')
    plt.ylabel('Average Returns')
    plt.title('DQN vs DoubleDQN vs uDQN on {}'.format(env_name))
    plt.legend(loc='upper left')
    plt.show()

    dqn_frames_list = list(range(len(dqn_max_q_value_list[0])))
    ddqn_frames_list = list(range(len(ddqn_max_q_value_list[0])))
    
    mean_dqn_max_q_value_list = np.mean(dqn_max_q_value_list, axis=0)
    mean_ddqn_max_q_value_list = np.mean(ddqn_max_q_value_list, axis=0)


    udqn_frames_list = list(range(len(udqn_max_q_value_list[0])))
    mean_udqn_max_q_value_list = np.mean(udqn_max_q_value_list, axis=0)



    plt.plot(dqn_frames_list, mean_dqn_max_q_value_list, label='DQN')
    plt.plot(ddqn_frames_list, mean_ddqn_max_q_value_list, label='doubleDQN')



    plt.plot(udqn_frames_list, mean_udqn_max_q_value_list, label='uDQN')
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Average Q value')
    plt.title('DQN vs DoubleDQN vs uDQN on {}'.format(env_name))
    plt.legend(loc='upper left')
    plt.show()
