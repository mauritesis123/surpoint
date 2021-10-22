# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:26:33 2021

@author: ramir
"""
from emulator import *
# cum_rewards = [np.nan] * env_t         ## recompensa acumulada

# def find_ideal(p, just_once):     ## ?? just_once
# 	if not just_once:
# 		diff = np.array(p[1:]) - np.array(p[:-1])
# 		return sum(np.maximum(np.zeros(diff.shape), diff))
# 	else:
# 		best = 0.
# 		i0_best = None
# 		for i in range(len(p)-1):
# 			best = max(best, max(p[i+1:]) - p[i])

# 		return best
   
   
   
   
   
# cum_rewards=0

# no_episodes =100




# def get_valid_actions(self):
#         if self.isAvailable:
#             return [0, 1]	# don't buy, buy
#         else:
#             return [0, 2]	# sell , hold

# def get_state(self, t=None):
        
#         state = self.sample_2d[self.current_index - self.last_n_timesteps + 1: self.current_index + 1, :].copy()

#         for i in range(state.shape[1]):
#             norm = np.mean(state[:,i])
#             state[:,i] = (state[:,i]/norm - 1.)*100

#         return state


# def reset(self, rand_price=True):
#         self.isAvailable = True

#         sample_2d = self.sampler.sample()
#         sample_1d = np.reshape(sample_2d[:,0], sample_2d.shape[0])

#         self.sample_2d = sample_2d.copy()
#         self.normalized_values = sample_1d/sample_1d[0]*100                    ## normalizar precio
#         self.last_index = self.normalized_values.shape[0] - 1

#         self.max_profit = find_ideal(self.normalized_values[self.start_index:], False)             ## 
#         self.current_index = self.start_index

#         return self.get_state(), self.get_valid_actions()


# def step(self, action):

#         if action == 0:		# don't buy / sell
#             reward = 0.
#             self.isAvailable = True
#         elif action == 1:	# buy
#             reward = self.get_noncash_reward()
#             self.isAvailable = False
#         elif action == 2:	# hold
#             reward = self.get_noncash_reward()
#         else:
#             raise ValueError('no such action: '+str(action))

#         self.current_index += 1

#         return self.get_state(), reward, self.current_index == self.last_index, self.get_valid_actions()

class moneda:
    ## INSTANCIAS
    def __init__(self, env):
        self.env = env
        
        
    def accion(self,state,valid_actions):
       if np.random.random() > 0.5:
          
          
       
       
 
   
 

    def act(self, state, valid_actions):
        if np.random.random() > self.epsilon:
            q_valid = self._get_q_valid(state, valid_actions)        ## lista de valores Q de la accion escogida
            if np.nanmin(q_valid) != np.nanmax(q_valid):             ## se busca accion que maximice Q
                return np.nanargmax(q_valid)                               
        return random.sample(valid_actions, 1)[0]  
     
      
     
    def _get_q_valid(self, state, valid_actions):
        q = self.model.predict(state)[0]                  ## predecir Q                                
        q_valid = [np.nan] * len(q)                       ## lista de valores Q para  acciones validas
        for action in valid_actions:
            q_valid[action] = q[action]
        return q_valid       
     
      
     
      
     def recompensa(self,state,valid_actions):
         cum_reward = env.step(i)[1]







   
for i in range(1,no_episodes+1):
   x=i;
   if  sampler.sample()[i]>sampler.sample()[i-1]:
       cum_rewards=
   
   
   
   