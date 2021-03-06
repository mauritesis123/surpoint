import os as os
os.chdir('C:/Users/ramir/OneDrive/Desktop/Surpoint/Proyecto 1/DQN-Trader-master')
from emulator import Market
from model import Q_Model
from agent import Agent
from sequence_generator import *
from simulator import Simulator

sampler = Single_Signal_Generator(total_timesteps=180, period_range=(10, 40), amplitude_range=(5, 80), noise_amplitude_ratio=0.5)
filename = "Generated Signals.npy"
#Sampler = PairSampler


# model_type = 'MLP'
# fld = os.path.join('..','data')
# sampler2 = Sampler('load', fld)




# sampler2 = PairSampler
# sampler.build_signals(filename2, 1000)
# sampler.load(filename2)


# sampler.build_signals(filename, 1000)
# sampler.load(filename)

env = Market(sampler=sampler, last_n_timesteps=40, buy_cost=3.3)


dense_model = [
    {"type":"Reshape", "target_shape":(env.get_state().shape[0]*env.get_state().shape[1],)},
    {"type":"Dense", "units":30},   
    {"type":"Dense", "units":30}
]
conv_model = [
    {"type":"Reshape", "target_shape":env.get_state().shape},
    {"type":"Conv1D", "filters":16, "kernel_size":3, "activation":"relu"},
    {"type":"Conv1D", "filters":16, "kernel_size":3, "activation":"relu"},
    {"type":"Flatten"},
    {"type":"Dense", "units":48, "activation":"relu"},
    {"type":"Dense", "units":24, "activation":"relu"}
]
gru_model = [
    {"type":"Reshape", "target_shape":env.get_state().shape},
    {"type":"GRU", "units":16, "return_sequences":True},
    {"type":"GRU", "units":16, "return_sequences":False},
    {"type":"Dense", "units":16, "activation":"relu"},
    {"type":"Dense", "units":16, "activation":"relu"}
]
lstm_model = [
    {"type":"Reshape", "target_shape":env.get_state().shape},
    {"type":"LSTM", "units":16, "return_sequences":True},
    {"type":"LSTM", "units":16, "return_sequences":False},
    {"type":"Dense", "units":16, "activation":"relu"},
    {"type":"Dense", "units":16, "activation":"relu"}
]





q_model = Q_Model("gru", state_dim=env.get_state().shape, no_of_actions=env.no_of_actions, layers=gru_model, hyperparameters={"lr":0.0001})
#q_model= Q_Model (lstm_model)

agent = Agent(q_model, batch_size=8, discount_factor=0.8, epsilon=1)

no_of_episodes_train = 1
no_of_episodes_test = 1

sim = Simulator(env, agent)
sim.train(no_of_episodes_train, epsilon_decay=0.997)




agent.model.save()
sim.test(no_of_episodes_test)




