import numpy as np
import random
import json
import os

np.random.seed(1)
random.seed(1)




class Single_Signal_Generator:

    ## INSTANCIAS 
    def __init__(self, total_timesteps, period_range, amplitude_range, noise_amplitude_ratio, sample_type="multi_sin_concat_with_base_whp",base_period_ratio=(2, 4), base_amplitude_range=(20, 80)):

        self.total_timesteps = total_timesteps              ## =180
        self.noise_amplitude_ratio = noise_amplitude_ratio  ## rango para amplitud del ruido
        self.period_range = period_range                    ## rango para el periodo
        self.amplitude_range = amplitude_range              ## amplitud de la señal precio
        self.base_period_range = (base_period_ratio[0]*period_range[1], base_period_ratio[1]*period_range[1])
        self.base_amplitude_range = base_amplitude_range
        self.base = False
        self.half_period = False
        self.loaded = False
        self.loaded_signals = None
        self.no_of_loaded_signals = None
        self.counter = 0



    ## GENERA LAS  SEÑALES
    def build_signals(self, filename, no_of_samples, sample_type="multi_sin_concat_with_base_whp", full_episode=True):
        if self.loaded:
            print("Signals already loaded!")
            return None

        signals = np.zeros((self.total_timesteps, no_of_samples))
        
        for i in range(no_of_samples):
            signals[:, i] = self.sample().reshape((signals.shape[0]))

        np.save(filename, signals)


    ## CARGAR
    def load(self, filename):
        self.loaded_signals = np.load(filename)
        self.loaded = True
        self.no_of_loaded_signals = self.loaded_signals.shape[1]


    ## TIPOS DE SEÑALES
    def sample(self, sample_type="multi_sin_concat_with_base_whp", full_episode=True):

        if self.loaded:
            if self.counter == self.no_of_loaded_signals:
                print("All loaded signals returned. Starting from the first signal again.")
                self.counter = 0

            return self.loaded_signals[:, self.counter].reshape((self.total_timesteps, 1))
        else:
            if sample_type == "multi_sin_concat_with_base_whp":
                self.base = True
                self.half_period = True
                return self._sample_multi_sin()
            elif sample_type == "single_sin":
                self.base = False
                self.half_period = False
                return self._sample_single_sin()
            elif sample_type == "multi_sin_concat":
                return self._sample_multi_sin()
            elif sample_type == "multi_sin_concat_whp":
                self.half_period = True
                return self._sample_multi_sin()
            else:
                print("Cannot recognise type. Defaulting to 'multi_sin_concat_with_base_whp'.")
                self.half_period = True
                return self._sample_multi_sin()

        

    def _random_sin(self, base=False, full_episode=False):

        if base:
            period = random.randrange(self.base_period_range[0], self.base_period_range[1])
            amplitude = random.randrange(self.base_amplitude_range[0], self.base_amplitude_range[1])
            noise = 0
        else:
            period = random.randrange(self.period_range[0], self.period_range[1])
            amplitude = random.randrange(self.amplitude_range[0], self.amplitude_range[1])
            noise = self.noise_amplitude_ratio * amplitude

        if full_episode:
            length = self.total_timesteps
        else:
            if self.half_period:
                length = int(random.randrange(1,4) * 0.5 * period)
            else:
                length = period

        signal_value = 100. + amplitude * np.sin(np.array(range(length)) * 2 * 3.1416 / period)
        signal_value += np.random.random(signal_value.shape) * noise

        return signal_value



    def _sample_single_sin(self):
        sample_container = []

        sample, function_name = self._random_sin(full_episode=True)

        sample_container.append(sample)

        return np.array(sample_container).T



    def _sample_multi_sin(self):
        sample_container = []
        sample = []
        while True:
            sample = np.append(sample, self._random_sin(full_episode=False)[0])
            if len(sample) > self.total_timesteps:
                break

        if self.base:
            base = self._random_sin(base=True, full_episode=True)
            sample_container.append(sample[:self.total_timesteps] + base)
            return np.array(sample_container).T
        else:
            sample_container.append(sample[:self.total_timesteps])
            return np.array(sample_container).T










class Sampler:
   def load_db(self, fld):
        #self.loaded_signals = np.load(fld)
        self.db = np.load(open(os.path.join(fld, 'db.npy')),'rb')
        self.loaded = True
        self.no_of_loaded_signals = self.loaded_signals.shape[1]   
        param = json.load(open(os.path.join(fld, 'param.json'),'rb'))
        self.i_db = 0
        self.n_db = param['n_episodes']
        self.sample = self.__sample_db
        for attr in param:                                    #attrs = ['title', 'window_episode', 'forecast_horizon_range', 'max_change_perc', 'noise_level', 'n_section', 'n_var']
           if hasattr(self, attr):
               setattr(self, attr, param[attr])
           self.title = 'DB_'+param['title']

    
   def build_db(self, n_episodes, fld):
        db = []
        #db= np.zeros((self.total_timesteps, no_of_samples))
        for i in range(n_episodes):
            prices, title = self.sample()
            db.append((prices, '[%i]_'%i+title))
        db=np.array(db)
        os.makedirs(fld)	# don't overwrite existing fld   ## DIRECCION DE LA BD
        np.save(os.path.join(fld, 'db.npy'), db)
		#pickle.dump(db, open(os.path.join(fld, 'db.pickle'),'wb'))
        param = {'n_episodes':n_episodes}
        for k in self.attrs:
           param[k] = getattr(self, k)
        json.dump(param, open(os.path.join(fld, 'param.json'),'w'))
        self.db=db


   def __sample_db(self):
         prices, title = self.db[self.i_db]
         self.i_db += 1
         if self.i_db == self.n_db:
             self.i_db = 0
         return prices, title
        
        
        
        
           
           
   








class PairSampler(Sampler):
    
   def __init__(self, game,window_episode=None, forecast_horizon_range=None, max_change_perc=10., noise_level=10., n_section=1,fld=None, windows_transform=[]):
         self.window_episode = window_episode
         self.forecast_horizon_range = forecast_horizon_range
         self.max_change_perc = max_change_perc
         self.noise_level = noise_level
         self.n_section = n_section
         self.windows_transform = windows_transform
         self.n_var = 2 + len(self.windows_transform) # price, signal
         self.attrs = ['title', 'window_episode', 'forecast_horizon_range', 
 			'max_change_perc', 'noise_level', 'n_section', 'n_var']
         param_str = str((self.noise_level, self.forecast_horizon_range, self.n_section, self.windows_transform))


         if game == 'load':
            self.load_db(fld)
         elif game in ['randwalk','randjump']:
             self.__rand = getattr(self, '_PairSampler__'+game)
             self.sample = self._sample
             self.title = game + param_str
         else:
             raise ValueError
 
   # def load(self, filename):
   #      self.loaded_signals = np.load(filename)
   #      self.loaded = True
   #      self.no_of_loaded_signals = self.loaded_signals.shape[1]   
        

  

   def __randwalk(self, l):
        change = (np.random.random(l + self.forecast_horizon_range[1]) - 0.5) * 2 * self.max_change_perc/100
        forecast_horizon = random.randrange(self.forecast_horizon_range[0], self.forecast_horizon_range[1])
        return change[:l], change[forecast_horizon: forecast_horizon + l], forecast_horizon

   def __randjump(self, l):
       change = [0.] * (l + self.forecast_horizon_range[1])
       n_jump = random.randrange(15,30)
       for i in range(n_jump):
           t = random.randrange(len(change))
           change[t] = (np.random.random() - 0.5) * 2 * self.max_change_perc/100
       forecast_horizon = random.randrange(self.forecast_horizon_range[0], self.forecast_horizon_range[1])
       return change[:l], change[forecast_horizon: forecast_horizon + l], forecast_horizon






   def _sample(self):

         L = self.window_episode
         if bool(self.windows_transform):
             L += max(self.windows_transform)
         l0 = L/self.n_section
         l1 = L

         d_price = []
         d_signal = []
         forecast_horizon = []

         for i in range(self.n_section):
             if i == self.n_section - 1:
                 l = l1
             else:
                 l = l0
                 l1 -= l0
             d_price_i, d_signal_i, horizon_i = self.__rand(l)
             d_price = np.append(d_price, d_price_i)
             d_signal = np.append(d_signal, d_signal_i)
             forecast_horizon.append(horizon_i)

         price = 100. * (1. + np.cumsum(d_price))
         signal = 100. * (1. + np.cumsum(d_signal)) + \
				np.random.random(len(price)) * self.noise_level

         price += (100 - min(price))
         signal += (100 - min(signal))

         inputs = [price[-self.window_episode:], signal[-self.window_episode:]]
         for w in self.windows_transform:
             inputs.append(signal[-self.window_episode - w: -w])

         return np.array(inputs).T, 'forecast_horizon='+str(forecast_horizon)
     
  
     
    
    

		


    # def build_signals(self, filename, no_of_samples, sample_type="multi_sin_concat_with_base_whp", full_episode=True):
    #     if self.loaded:
    #         print("Signals already loaded!")
    #         return None
    #     signals = np.zeros((self.total_timesteps, no_of_samples))
        
    #     for i in range(no_of_samples):
    #         signals[:, i] = self.sample().reshape((signals.shape[0]))

    #     np.save(filename, signals)







     



if __name__ == '__main__':
     gen = Single_Signal_Generator(180, (10, 40), (5, 80), 0.5)

     filename = "Generated Signals.npy"
     #filename2 = "signal.npy"
     gen.build_signals(filename, 1000)
     gen.load(filename)
     print(gen.sample())
     print(gen.loaded_signals.shape)








