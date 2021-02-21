# ***
# tienes que refactorizar esta vaina porque no se sabe si verdaderamente sirve esta porqueria y ta escrito bien feo
# tambien hay que agregar cosas en el env para que sea mas realistico (como el tax cuando compras/vendes)
# hay una vaina rara con las dimensiones de la observacion asi que tendras que revisar eso tambien para ver si sirve bien.
# ***
import numpy as np
import random

class environment:
    # you pass the historical data as a numpy array to the init.
    # first you have to save the price at the time of the action.  
    # 
    
    def __init__(self, historical_data, observation_size):
        self.historical_data = historical_data
        
        self.leverage = 10
        self.action_history = []
        self.buy_sell = None # variable to know if the first position is either a buy or a sell; buy = 0, sell = 1
        self.current_price = None
        self.initial_price = None
        self.reward = 0
        self.timesteps = 0
        self.observation = None
        self.info = None # unused as of now
        self.done = None
        self.current_position = None
        self.start_position = None
        self.observation_size = observation_size
        self.first_run = True

        self.buy_sell_price_obs = np.array([0, 0], dtype = np.float32)
        self.full_observation = []
        

        
    def step(self, action):
        # get action and return what the agent needs. take into account the penalty for instantly buying and selling.
        # as of now I plan to do it as an episodic task, I will have to check this approach as it could bring problems later.
        # episode ends when agent gets a return(ie, buys then sells or sells then buys)
        # reward is the profit that the agent gets at the end of the episode
        # info might be how many timesteps the episode lasted
        # think about implementing variable reward or reward decaying to get certain behaviours early on.
        # maybe we should penalize consecutive buys and consecutive sells?
        # maybe the agent should know if he's bought or sold previuosly in this episode? (almost definitely) (encoding/0,1,2)
        # the agent should also know the price that he bought at. (as an input to the neural network) 
        # now that I think about it, the reward cant be just profit, it has to be the ratio of investment to return so that the
        # reward remains constant.
        self.first_run = False
        self.reward = 0
        
        
        self.timesteps = self.timesteps + 1
        self.current_position = self.start_position + self.timesteps 
        self.observation = self.historical_data[self.current_position - self.observation_size : self.current_position]
        self.current_price = self.observation[self.observation_size - 1, 3] #get current price based on current position
        
        
        if action == 0:
            # you dont actually have to do much
            self.done = False
            self.reward = 0
            
        if action == 1:
            self.done = False
            self.reward = 0
            
            #first buy
            if self.buy_sell == None:
                self.buy_sell = 0
                self.buy_sell_price_obs[0] = 1
                self.buy_sell_price_obs[1] = self.current_price
            
            # buy buy
            if self.buy_sell == 0:
                self.action_history.append(self.initial_price)
            # sell buy    
            if self.buy_sell == 1:
                #get all prices and get the total profit
                # remember that the calculation is done in the opposite side. so this calculation is for selling then buying
                # and the return is positive when the current price is lower than the initial price
                for price in self.action_history:
                    self.reward = self.reward + (price * self.leverage) - (self.current_price * self.leverage)
                
                self.done = True
                print("The episode ran for {} timesteps.".format(self.timesteps))
                self.observation = self.reset()
                
            
        if action == 2:
            self.done = False
            self.reward = 0
            
            #first sell
            if self.buy_sell == None:
                self.buy_sell = 1
                self.buy_sell_price_obs[0] = 2
                self.buy_sell_price_obs[1] = self.current_price
                
            # sell sell
            if self.buy_sell == 1:
                self.action_history.append(self.initial_price)
            # buy sell
            if self.buy_sell == 0:
                # get all prices and get total profit
                # remember that the calculation is done in the opposite side. so this calculation is for buying then selling
                # and the return is positive when the current price is higher than the initial price
                for price in self.action_history:
                    self.reward = self.reward - (price * self.leverage) + (self.current_price * self.leverage)
                
                self.done = True
                print("The episode ran for {} timesteps.".format(self.timesteps))
                self.observation = self.reset()
                
            
        # full_observations is a [2] list which has the price action observation [observation_size, 5]
        # and buy_sell_price_obs [2] con la memoria de la accion y el precio al momento de la accion. 
        
        self.observation_full = []
        self.full_observation = [self.observation, self.buy_sell_price_obs]
#         print(self.full_observation)
        return self.full_observation, self.reward, self.done, self.info
    
    def reset(self):
        # get random starting point in historical data and return the first point as an observation.
        # reset all variables used for the episode
        
        # you have to check more thoroughly to determine which variables you dont have to reset
        # this is so that you can return properly on the env.step() function.
        
        self.action_history = []
        self.buy_sell = None # variable to know if the first position is either a buy or a sell; buy = 0, sell = 1
        self.current_price = None
        self.initial_price = None
#         if self.first_run:
#             print("The episode ran for {} timesteps.".format(self.timesteps))
        self.timesteps = 0
        self.observation = None
        self.current_position = None
        self.start_position = None
        self.first_price_action = 0
        
        # test
        self.buy_sell_price_obs = np.array([0, 0], dtype = np.float32)
        self.observation_full = []
        
        
        self.start_position = random.randint(self.observation_size, 45000) # get random number to start
#         print(self.start_position)
        observation = self.historical_data[self.start_position - self.observation_size : self.start_position]
        self.initial_price = observation[self.observation_size - 1, 3]

        if self.first_run:
            self.full_observation = [observation, self.buy_sell_price_obs]
            return self.full_observation
        self.first_run = True
        return observation