import numpy as np
import random

class Agent:
    
    ## INSTANCIAS: modelo, batch_size, discount_factor, epsilon
    def __init__(self, model, batch_size, discount_factor, epsilon):
        self.model = model                                       ## modelo
        self.batch_size = batch_size                             ## tamaño de muestras de datos 
        self.discount_factor = discount_factor                   ## factor de descuento gamma
        self.memory = []                                         ## memoria
        self.epsilon = epsilon                                   ## probabilidad de seleccionar accion



    ## OBTENCION DE VALORS Q PARA PARES ESTADO-ACCION
    def _get_q_valid(self, state, valid_actions):
        q = self.model.predict(state)[0]                  ## predecir Q                                
        q_valid = [np.nan] * len(q)                       ## lista de valores Q para  acciones validas
        for action in valid_actions:
            q_valid[action] = q[action]
        return q_valid                                    ## Retorna lista de valores Q 



    ## ACCION  SE ESCOGE CON PROB= EPSILON
    def act(self, state, valid_actions):
        if np.random.random() > self.epsilon:
            q_valid = self._get_q_valid(state, valid_actions)        ## lista de valores Q de la accion escogida
            if np.nanmin(q_valid) != np.nanmax(q_valid):             ## se busca accion que maximice Q
                return np.nanargmax(q_valid)                               
        return random.sample(valid_actions, 1)[0]                    ## devuelve accion de Qmax



    ## ALMACENAMIENTO DE LA EXPERIENCIA=[(state, action, reward, next_state, done, valid_actions)]
    def remember(self, experience):
        self.memory.append(experience)                               ## concatenar experiencia



    ## REPETICION ALEATORIA DE EXPERIENCIA
    def replay(self):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))                          ## muestra aleatoria de la memoria, de tamaño min(len(self.memory), self.batch_size)
        for state, action, reward, next_state, done, next_valid_actions in batch:
            q = reward
            if not done:
                q += self.discount_factor * np.nanmax(self._get_q_valid(next_state, next_valid_actions))    ##  Q= r + gamma * MAX Q(s_{j+1}, a) 
            self.model.fit(state, action, q)



    ## GUARDAR MODELO
    def save_model(self):
        self.model.save()
