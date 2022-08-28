import gym
import numpy as np
import torch

class TrainManager():
    def __init__(self,env,episodes=1000,lr=0.001,gamma=0.9,e_greed=0.1) -> None:
        self.env = env
        self.episodes = episodes
        self.lr = lr
        self.gamma = gamma
        self.e_greed = e_greed
class Agent():
    def __init__(self , q_func ,optimizer, n_action , gamma = 0.9,lr = 0.1 ,e_greed = 0.1):
      
        self.q_func = q_func
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optimizer

        self.n_action = n_action
        self.lr = lr
        self.gamma = gamma
        self.e_greed = e_greed

        self.criterion = torch.nn.MSELoss()
        self.optimizer = optimizer
    def predict(self,obs):
        Q_list = self.q_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action
    
    def act(self,obs):
        if np.random.uniform(0,1) < self.e_greed:
            action = np.random.choice(self.n_action)
        else :
            action = self.predict(obs)
        return action

    def learn(self,obs,action,reward,obs_next,action_next,done):
        cur_Q = self.q_func(obs)[action]
        target_Q = reward + (1-float(done))*self.gamma * self.q_func(obs_next).max()
        loss = self.criterion(cur_Q,target_Q)
        loss.backward()
        self.optimizer.step()


