import gym
import numpy as np
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
        # sarsa
        # target_Q = reward + (1-float(done))*self.gamma * self.Q[state_next, action_next]
        #Q-learning
        target_Q = reward + (1-float(done))*self.gamma * self.Q[state_next, :].max()
        self.Q[state,action] += self.lr * (target_Q - cur_Q)

def train_episode(env,agent,is_render): 
    reward_total = 0
    state  = env.reset()
    action = agent.act(state) 
    done = False
    while not done:
        if is_render:
            env.render()
        state_next , reward, done, info = env.step(action)
        action_next = agent.act(state_next)
        agent.learn(state,action,reward,state_next,action_next,done)
        state  = state_next
        action = action_next 
        reward_total += reward
    return reward_total

def test_episode(env,agent,is_render):
    reward_total = 0
    state  = env.reset()
    action = agent.predict(state) 
    done = False
    while not done:
        if is_render:
            env.render()
        state_next ,reward ,done ,info= env.step(action)
        action_next = agent.predict(state_next)
        agent.learn(state,action,reward,state_next,action_next,done)
        state  = state_next
        action = action_next 
        reward_total += reward
    return reward_total
def train(env,episodes = 1000 ,e_greed = 0.1,lr=0.1,gamma = 0.9):
    agent = SarsaAgent(
        n_states = env.observation_space.n,
        n_action    = env.action_space.n)
    is_render = False
    for e in range(episodes):
        ep_reward = test_episode(env, agent,is_render)
        print('Episode %s: reward = %.1f' % (e, ep_reward))
        if e %200== 99:
            is_render = True
        else :
            is_render = False
    
if __name__ == '__main__' :
    env = gym.make("CliffWalking-v0")
    train(env)
