from ast import Not
from sys import float_repr_style
import numpy as np
import gym

class SarsaAgent():
    def __init__(self , n_states , n_action , gamma = 0.9,lr = 0.1 ,e_greed = 0.1):
        self.Q = np.zeros((n_states , n_action))
        self.n_action = n_action
        self.lr = lr
        self.gamma = gamma
        self.e_greed = e_greed
        self.n_status = n_states
    def predict(self,state):
        Q_list = self.Q[state,:]
        action = np.random.choice(np.flatnonzero(Q_list==Q_list.max()))
        return action
    
    def act(self,state):
        if np.random.uniform(0,1) < self.e_greed:
            action = np.random.choice(self.n_action)
        else :
            action = self.predict(state)
        return action

    def learn(self,state,action,reward,state_next,action_next,done):
        cur_Q = self.Q[state,action]
        if done:
            target_Q = reward
        else :
            target_Q = reward + self.gamma * self.Q[state_next, action_next]
        self.Q[state,action] += self.lr * (target_Q - cur_Q)

def train_episode(env,agent):
    state  = env.reset()
    action = agent.act(state) 
    done = False
    while not done:
        state_next,reward,done,info = env.step(action)
        action_next = agent.act(state_next)
        agent.learn(state,action,reward,state_next,action_next,done)
        state  = state_next
        action = action_next 
      
def test_episodes():
   pass

def train(env,episodes = 500 ,e_greed = 0.1,lr=0.1,gamma = 0.9):
    agent = SarsaAgent(
        n_states = env.observation_space.n,
        n_action    = env.action_space.n)
    for e in range(episodes):
        ep_reward = train_episode(env, agent)
    #   print('Episode %s: reward = %.1f' % (e, ep_reward))


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0", render_mode='human')  # 0上, 1右, 2下, 3左
    train(env)