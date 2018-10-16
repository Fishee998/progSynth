import torch
import torch.nn as nn
from torch.autograd import Variable

#import tensorflow as tf
import numpy as np
import gym
from MLP import MLP
from gym import wrappers
import math
import astEncoder
import example
from random import choice
import StringIO


class DQN():
    def __init__(self, env, alpha, gamma, episode_num, target_reward, step_count, minbatch, memory_size, flag):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.episode_num = episode_num
        self.target_reward = target_reward
        self.step_count = step_count
        # self.test_step=test_step
        self.minbatch = minbatch
        self.memory_size = memory_size
        self.flag = flag
        self.Q = MLP()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.spaces[0].n * env.action_space.spaces[1].n
        # self.action_dim = env.action_space.n

        self.Q.creat2(self.state_dim, env.action_space.spaces[0].n, env.action_space.spaces[1].n)

        self.memory_num = 0
        self.memory = np.zeros((memory_size, self.state_dim * 2 + 4))
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)
        self.loss_func = nn.MSELoss()

    # action1 action2 legal
    def action2set(self, action2s):
        actionLen =example.action2Len(action2s)
        action2Seleced = []
        for index in range(actionLen):
            action2Seleced.append(example.get_action2(action2s, index))
        return action2Seleced

    def getAction1(self, act1Set, action1_value):
        nonzeroind = np.nonzero(act1Set)[0]
        index = torch.LongTensor([nonzeroind])
        action1_values = torch.gather(action1_value.data, 1, index)
        action1 = torch.max(Variable(action1_values), 1)[1].data.numpy()[0]
        action1 = nonzeroind[action1]
        return action1

    def getAction1_random(self, act1Set):
        act1Set1 = np.nonzero(act1Set)
        action1 = choice(act1Set1[0])
        return action1

    def getAction2(self, candidate, action1, action2_value):
        action2 = example.getLegalAction2(candidate, action1)
        action2set_ = self.action2set(action2)
        action2_ = []
        for index2 in range(50):
            action2_.append(0)
        for index in range(len(action2set_)):
            a = action2set_[index]
            action2_[a] = 1
        nonzeroind2 = np.nonzero(action2_)[0]
        index2 = torch.LongTensor([nonzeroind2])
        action2_values = torch.gather(action2_value.data, 1, index2)
        action2 = torch.max(Variable(action2_values), 1)[1].data.numpy()[0]
        action2 = nonzeroind2[action2]
        return action2

    def getAction2_random(self, candidate, action1):
        action2_ = example.getLegalAction2(candidate, action1)
        action2set = self.action2set(action2_)
        action2 = choice(action2set)
        return action2

    def getAction(self, action1, action2):
        action = []
        action.append(action1)
        action.append(action2)
        action = tuple(action)
        return action

    def choose_action(self, state,episode, act1Set, candidate):
        # epsilon = 0.5 * (0.993) ** episode
        epsilon = 0.8 * (0.993) ** episode
        if epsilon < 0.3:
            epsilon = 0.3
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))

        if np.random.uniform() > epsilon:
            # print("action by rl")
            action1_value, action2_value = self.Q.forward(state)
            action1 = self.getAction1(act1Set, action1_value)
            action2 = self.getAction2(candidate, action1, action2_value)
            action = self.getAction(action1, action2)
        else:
            # print("action randomly")
            action1 = self.getAction1_random(act1Set)
            action2 = self.getAction2_random(candidate, action1)
            action = self.getAction(action1, action2)

        return action

    def select_action(self, state, act1Set, candidate):
        '''
        action = np.random.randint(0, self.action_dim)
        '''
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        action1_value, action2_value = self.Q.forward(state)
        action1 = self.getAction1(act1Set, action1_value)
        action2 = self.getAction2(candidate, action1, action2_value)
        action = self.getAction(action1, action2)
        return action

    def store_transition(self, state, action0, action1, reward, done,  next_state):
        transition = np.hstack((state, [action0, action1, reward, done], next_state))
        index = self.memory_num % self.memory_size
        self.memory[index, :] = transition
        self.memory_num += 1

    def learn(self):

        sample = np.random.choice(self.memory_size, self.minbatch)
        batch = self.memory[sample, :]
        state_batch = Variable(torch.FloatTensor(batch[:, :self.state_dim]))
        action1_batch = Variable(torch.LongTensor(batch[:, self.state_dim:self.state_dim+1].astype(int)))
        action2_batch = Variable(torch.LongTensor(batch[:, self.state_dim+1:self.state_dim + 2].astype(int)))
        reward_batch = Variable(torch.FloatTensor(batch[:, self.state_dim+2:self.state_dim+3]))
        done_batch = Variable(torch.FloatTensor(batch[:, self.state_dim+3:self.state_dim+4].astype(int)))
        next_state_batch = Variable(torch.FloatTensor(batch[:, -self.state_dim:]))

        # q = self.Q(state_batch).gather(1, action_batch)
        q1 = self.Q(state_batch)[0].gather(1, action1_batch)
        q2 = self.Q(state_batch)[1].gather(1, action2_batch)
        q1_next = self.Q(next_state_batch)[0].detach()
        q2_next = self.Q(next_state_batch)[1].detach()
        q1_val = q1_next.max(1)[0].view(self.minbatch, 1)
        q2_val = q2_next.max(1)[0].view(self.minbatch, 1)
        if self.flag == 0:
            for i in range(len(done_batch)):
                if done_batch[i].data[0] == 1:
                    q1_val[i] = 0
        y1 = reward_batch + self.gamma * q1_val
        loss1 = self.loss_func(q1, y1)


        y2 = reward_batch + self.gamma * q2_val
        loss2 = self.loss_func(q2, y2)
        loss = loss1 + loss2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        result = loss.data[0]
        return result

    def train_CartPole(self,label):
        loss=[]
        buf = StringIO.StringIO()
        print("error1")
        # buf2 = StringIO.StringIO()
        buf3 = StringIO.StringIO()
        state_init, info_init = self.env.reset()
        for i_episode in range(self.episode_num):
            ep_r = 0
            #if i_episode % 500 == 0:
            state_init, info_init = self.env.reset()
            state = state_init
            info_ = info_init
            print("new episode")
            actIndex = astEncoder.setAction1s(info_)
            loss_num = 0
            buf2 = StringIO.StringIO()
            # buf2.write("episode: %d" % (i_episode))
            for t in range(self.step_count):
                # env.render()
                action = self.choose_action(state, i_episode, actIndex, info_.candidate)
                next_state, reward, done, info_ = self.env.step(action)
                #if info_.fitness > 69:
                    #state_init = next_state
                    #info_init = info_
                actIndex = astEncoder.setAction1s(info_)
                # spin_reward = example.spin_(info_.candidate)
                if done and example.get_fitness(info_.candidate) > 78.4:
                    reward = self.target_reward
                    print("i_ep" + str(i_episode) + " step:" + str(t) + " fitness" + str(example.get_fitness(info_.candidate)))
                    spin_reward = example.spin_(info_.candidate)
                    # buf2.write("program at i_ep %d: step:%d  fitnessValue:%d\n" % (i_episode, t, example.get_fitness(info_.candidate)))
                    if spin_reward == 20:
                        buf.write("correct program at i_ep %d: step:%d \n" % (i_episode, t))
                        fo = open("./correctProg.txt", "a+")
                        fo.write(buf.getvalue())
                        fo.close()
                    if spin_reward == 5:
                        print("liveness")
                    if spin_reward == 10:
                        print("safety")
                    reward = reward + spin_reward

                reward += reward

                self.store_transition(state, action[0], action[1], reward, done, next_state)
                if self.memory_num > self.memory_size:
                    loss_num += self.learn()

                if done:
                    loss.append(loss_num / (t + 1))
                state = next_state
                if t % 100 == 0:
                    print("i_ep: ", i_episode, "step: ", t, "reward: ", ep_r)
                    #buf2.write("i_ep %d: step:%d reward: %f\n" % (i_episode, t, ep_r))
            fou = open("./rewards.txt", "a+")
            fou.write(buf2.getvalue())
            fou.close()
            # fou1 = open("/Users/zhuang/workspace-gp/testSwig2/record.txt", "a+")
            # fou1.write(buf3.getvalue())
            # fou1.close()

        # fo = open("/Users/zhuang/workspace-gp/testSwig2/foo.txt", "a+")
        # fo.write('state:\n' + str(state) + '\n' + 'action:\n' + str(action) + '\n' + 'reward:\n' + str(
            # reward) + '\n' + 'nextState:\n' + str(next_state) + '\n')
        # fo.close()





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
    # env,alpha,gamma,episode_num,target_reward,step_count,test_step,minbatch,memory_size,flag
    dqnCartPole = DQN(env,0.001,0.9,200000,80,2000,64,500,0)
    dqnCartPole.train_CartPole('CartPoleDQNTrain')
    # dqnCartPole.test('CartPoleDQNTest')


