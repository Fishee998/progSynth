import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gym
from MLP import MLP
from gym import wrappers
import matplotlib.pyplot as plt
import math
import astEncoder
import tensorflow as tf

#gym.make("My-progSynth")


class DQN():
    def __init__(self,env,alpha,gamma,episode_num,target_reward,step_count,test_step,minbatch,memory_size,flag):
        self.env=env
        self.alpha=alpha
        self.gamma=gamma
        self.episode_num=episode_num
        self.target_reward=target_reward
        self.step_count=step_count
        self.test_step=test_step
        self.minbatch=minbatch
        self.memory_size=memory_size
        self.flag=flag
        self.Q = MLP()
        # print(env.action_space.spaces[0].n * env.action_space.spaces[1].n)
        self.state_dim = env.observation_space.shape[0]

        self.action_dim = env.action_space.spaces[0].n * env.action_space.spaces[1].n
        # self.action_dim = env.action_space.n

        self.Q.creat2(self.state_dim, self.action_dim)

        self.memory_num = 0
        self.memory = np.zeros((memory_size, self.state_dim * 2 + 3))
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)
        self.loss_func = nn.MSELoss()

    def pic(self,x,y,label,suffix):
        x_min=np.min(x)
        x_max=np.max(x)
        y_min=np.min(y)
        y_max=np.max(y)
        plt.xlabel(label+' episode')
        plt.ylabel(label+suffix)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.plot(x,y)
        plt.savefig('./images/'+label+suffix+'.png', format='png')
        plt.close()

    def choose_action(self, state,episode):
        epsilon = 0.5 * (0.993) ** episode
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))

        if np.random.uniform() > epsilon:
            actions_value = self.Q.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def select_action(self, state, info_):
        act1set = astEncoder.setAction1s(info_)
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        actions_value = self.Q.forward(state)
        actions_value[0, 25*6:(76800 - 1)] = 0
        for index in range(len(act1set)):
            if index < 5 and act1set[index] == 0:               # layer2
                actions_value[0, index * 6: (index+1) * 6 - 1] = 0
            elif index > 14 and act1set[index] == 0:            # layer4
                actions_value[0, ((index - 15) * 6 + 30): ((index - 14) * 6 + 30 - 1)] = 0
            elif act1set[index] == 0:
                actions_value[0, (76800 + 76800*(index-5)): (76800 + 76800*(index-4) - 1)] = 0
        print(len(torch.nonzero(actions_value)))
        action = torch.max(actions_value, 1)[1].data.numpy()[0]
        if action / 76800 == 0:
            action1_ = 0
            nodenum = action / 6
            action2_ = action % 76800
            # actType = action % 6
            # if nodenum > 4:
                # nodenum = nodenum + 10
        else:
            action1_ = action / 76800
            action2_ = action % 76800
            # nodenum = info_.astActNodes[action / 76800 + 4] # dim 0 :0-4 and 25-34 dim 1-10: layer2 5-24
            # actType = action % 76800 - 1
        action = (action1_, action2_)
        return action

    def store_transition(self, state, action, reward, done,  next_state):
        transition = np.hstack((state, [action, reward, done], next_state))

        index = self.memory_num % self.memory_size
        self.memory[index, :] = transition
        self.memory_num += 1

    def learn(self):

        sample = np.random.choice(self.memory_size, self.minbatch)
        batch = self.memory[sample, :]
        state_batch = Variable(torch.FloatTensor(batch[:, :self.state_dim]))
        action_batch = Variable(torch.LongTensor(batch[:, self.state_dim:self.state_dim+1].astype(int)))
        reward_batch = Variable(torch.FloatTensor(batch[:, self.state_dim+1:self.state_dim+2]))
        done_batch = Variable(torch.FloatTensor(batch[:, self.state_dim+2:self.state_dim+3].astype(int)))
        next_state_batch = Variable(torch.FloatTensor(batch[:, -self.state_dim:]))

        q = self.Q(state_batch).gather(1, action_batch)
        q_next = self.Q(next_state_batch).detach()
        q_val = q_next.max(1)[0].view(self.minbatch, 1)
        if self.flag==0:
            for i in range(len(done_batch)):
                if done_batch[i].data[0]==1:
                    q_val[i] = 0
        y = reward_batch + self.gamma * q_val
        loss = self.loss_func(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        result = loss[0].data[0]
        return result

    def train(self,label):
        total_step = 0
        xa=[]
        ya=[]
        loss = []
        for i_episode in range(self.episode_num):
            xa.append(i_episode)
            state = self.env.reset()
            ep_r = 0
            loss_num = 0
            for t in range(self.step_count):
                env.render()
                action = self.choose_action(state,i_episode)
                next_state, reward, done, info = self.env.step(action)
                if self.flag==0:
                    reward+=0
                elif self.flag==1:
                    pos,vel=next_state
                    reward=abs(pos+0.5)
                    reward+=abs(next_state[0]-state[0])
                elif self.flag==2:
                    reward+=0

                if done:
                    reward=self.target_reward

                self.store_transition(state, action, reward, done, next_state)
                ep_r += reward
                if self.memory_num > self.memory_size:
                    loss_num+=self.learn()

                if t==(self.step_count-1):
                    total_step += t + 1
                    ya.append(ep_r)
                    loss.append(loss_num / (t + 1))
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2),'setp',t+1,'average step',total_step / (i_episode + 1))
                    break

                if done:
                    total_step += t + 1
                    ya.append(ep_r)
                    loss.append(loss_num / (t + 1))
                    print('Ep: ', i_episode,
                            '| Ep_r: ', round(ep_r, 2),'setp',t+1,'average step',total_step / (i_episode + 1))
                    break
                state = next_state
        self.pic(xa,ya,label,' Reward')
        self.pic(xa, loss, label, ' Loss')

    def train_CartPole(self,label):
        total_step = 0
        count = 0
        xa = []
        ya = []
        xloss=[]
        loss=[]
        for i_episode in range(self.episode_num):

            if i_episode % 10 == 0:
                sum = 0
                for i in range(100):
                    xa.append(count)
                    count += 1;
                    ep_r = 0
                    state, info_ = self.env.reset()
                    actIndex = astEncoder.setAction1s(info_)
                    for t in range(self.test_step):
                        # env.render()
                        action = self.select_action(state, info_)

                        next_state, reward, done, info_ = self.env.step(action)


                        print(reward)
                        ep_r += reward

                        if done or t == (self.test_step - 1):
                            sum += t + 1
                            ya.append(ep_r)
                            print('Ep: ', i,
                                  'setp', t + 1, 'average step', sum / (i + 1))
                            break
                        state = next_state
                if sum / 100 > (self.test_step / 2):
                    break
            if len(xa) != len(ya):
                print(len(xa), xa)
                print(len(ya), ya)
                exit(1)
            xa.append(count)
            xloss.append(i_episode+1)
            count += 1
            state = self.env.reset()
            ep_r = 0
            loss_num=0
            for t in range(self.step_count):
                # env.render()
                action = self.choose_action(state, i_episode)

                next_state, reward, done, info = self.env.step(action)

                if done:
                    reward = self.target_reward

                self.store_transition(state, action, reward, done, next_state)

                ep_r += reward
                if self.memory_num > self.memory_size:
                    loss_num+=self.learn()

                if t == (self.step_count - 1):
                    total_step += t + 1
                    ya.append(ep_r)
                    loss.append(loss_num/(t+1))
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2), 'setp', t + 1, 'average step', total_step / (i_episode + 1))
                    break

                if done:
                    total_step += t + 1
                    ya.append(ep_r)
                    loss.append(loss_num/(t+1))
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2), 'setp', t + 1, 'average step', total_step / (i_episode + 1))
                    break
                state = next_state

        self.pic(xa, ya, label,' Reward')
        self.pic(xloss,loss,label,' Loss')

    def test(self,label):
        total_step = 0
        x=[]
        y=[]

        total_reward = 0
        rlist = []

        for i_episode in range(1000):
            if i_episode==9999:
                self.env = wrappers.Monitor(self.env, './video/DQN/'+label)
            state = self.env.reset()
            i_reward = 0
            x.append(i_episode)
            for t in range(self.test_step):
                # self.env.render()
                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                i_reward += reward

                if t == (self.test_step-1):
                    total_step += t + 1
                    y.append(i_reward)

                    break

                if done:
                    y.append(i_reward)
                    total_step += t + 1
                    break
                state = next_state
            rlist.append(i_reward)
            total_reward += i_reward
            print('%d Episode finished after %f time steps' % (i_episode, t + 1))
            ar = total_reward / (i_episode + 1)
            print('average reward:', ar)
            av = total_reward / (i_episode + 1)
            sum = 0
            for count in range(len(rlist)):
                sum += (rlist[count] - av) ** 2
            sr = math.sqrt(sum / len(y))
            print('standard deviation:', sr)
        self.pic(x,y,label,'Reward')


if __name__ == "__main__":

    gym.envs.register(
        id='CartPoleExtraLong-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=20000,
        reward_threshold=19500.0,
    )

    env = gym.make('CartPoleExtraLong-v0')
    dqnCartPole = DQN(env,0.001,0.9,5000,-20,20000,2000,64,5000,0)
    dqnCartPole.train_CartPole('CartPoleDQNTrain')
    dqnCartPole.test('CartPoleDQNTest')

    gym.envs.register(
        id='MountainCarExtraLong-v0',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=2000,
        reward_threshold=19500.0,
    )
    env = gym.make('MountainCarExtraLong-v0')
    mountain_car_dqn = DQN(env, 0.001, 0.9, 1000, 2, 2000, 2000, 32, 2000,1)
    mountain_car_dqn.train('MountainCarDQNTrain')
    mountain_car_dqn.test('MountainCarDQNTest')

    gym.envs.register(
        id='AcrobotExtraLong-v1',
        entry_point='gym.envs.classic_control:AcrobotEnv',
        max_episode_steps=2000,
        reward_threshold=19500.0,
    )
    env = gym.make('AcrobotExtraLong-v1')
    acrobot_dqn = DQN(env, 0.01, 0.999, 500, 0.5, 2000, 2000,32, 5000,2)
    acrobot_dqn.train('AcrobotDQNTrain')
    acrobot_dqn.test('AcrobotDQNTest')
