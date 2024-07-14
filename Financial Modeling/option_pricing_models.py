# The stock tree will be represented using nodes (i, j) and initial stock price of S0
# Si,j = S0u^(j)d^(i-j)
# Ci,j = contract price at each node in the tree (i,j), where CNj = final payoff function that we will define
# Going to price European Call option, so CNj = max(SNj - K,0)

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.stats import norm

class BinomialOptionPricingModel():
    def __init__(self, S0, K, T, sigma, r, N, opttype='C'):
        self.asset_price = S0
        self.strike_price = K
        self.time_to_maturity = T
        self.asset_volatility = sigma
        self.risk_free_rate = r
        self.num_time_steps = N
        self.opttype = opttype
        self.stock_price_tree = []
        self.derivative_price_tree = []
    
    def calculate_derivative_price(self):
        # At each step in the tree, the price can move up or down by a factor of u or d respectively
        delta_t = self.time_to_maturity / self.num_time_steps
        u = np.exp(self.asset_volatility * np.sqrt(delta_t))
        d = np.exp(-1 * self.asset_volatility * np.sqrt(delta_t))
        # Calculating risk neutral probability
        P = ((np.exp(self.risk_free_rate*delta_t) - d)) / (u - d)

        
        for level_in_tree in range(1,self.num_time_steps+2):
            self.stock_price_tree.append([0 for _ in range(0, level_in_tree)])
            self.derivative_price_tree.append([0 for _ in range(0, level_in_tree)])
        
        # First need to calculate the initial values for the last layer of the tree

        for i in range(self.num_time_steps+1): # Tree looks like this [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
            # print(f"Our binomial tree is: {stock_price_tree}")
            # print(f"Length of the list nodes at index {i}: {len(stock_price_tree[i])}")
            for j in range(0, len(self.stock_price_tree[i])):
                # print(f"Our binomial tree is: {stock_price_tree}")
                calculation = self.asset_price * (u ** j) * (d ** i)
                self.stock_price_tree[i][j] = self.asset_price * (u ** j) * (d ** (i-j))

        for k in range(0, self.num_time_steps+1): # Tree looks like this [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
            if self.opttype == 'C':
                self.derivative_price_tree[-1][k] = max(self.stock_price_tree[-1][k] - self.strike_price, 0) # For call
            elif self.opttype == 'P':
                self.derivative_price_tree[-1][k] = max(self.strike_price - self.stock_price_tree[-1][k], 0) # For Put


        for x in range(self.num_time_steps-1, -1, -1):
            for y in range(len(self.derivative_price_tree[x])):
                self.derivative_price_tree[x][y] = np.exp(-1*self.risk_free_rate*delta_t) * (P * self.derivative_price_tree[x+1][y+1] + (1-P)*self.derivative_price_tree[x+1][y])
        # derivative_price_tree[k][n] = np.exp(-1*r*delta_t) * P * stock_price_tree[k-1][n-1] + (1-P)*derivative_price_tree[k-1][n]

        #stock_price_tree[i][j] = np.exp(-1*r*delta_t) * P * stock_price_tree[i-1][j-1] + (1-P)*stock_price_tree[i-1][j]


        return self.derivative_price_tree[0][0]

# print(calculate_derivative_price(S0, K, T, sigma, r, N))

volatility_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
r_list =          [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

option_prices = np.zeros((len(volatility_list), len(r_list)))

binomial_model = BinomialOptionPricingModel(100, 100, 1, 0.2, 0.05, 5, 'C')
print(binomial_model.calculate_derivative_price())
# for i, sigma in enumerate(volatility_list):
#     for j, r in enumerate(r_list):
#         to_set = binomial_option_pricing_model(S0, K, T, sigma, r, N)
#         print(to_set)
#         option_prices[i,j] = to_set




# plt.figure(figsize=(10, 8))
# plt.imshow(option_prices, extent=[r_list[0], r_list[-1], volatility_list[0], volatility_list[-1]], origin='lower', aspect='auto', cmap='viridis')
# plt.colorbar(label='Option Price')
# plt.xlabel('Risk-Free Rate')
# plt.ylabel('Volatility')
# plt.title('Heatmap of Option Prices')
# plt.grid(True, linestyle='--', linewidth=0.5)

# # Add contour lines
# contours = plt.contour(option_prices, colors='white', origin='lower', extent=[r_list[0], r_list[-1], volatility_list[0], volatility_list[-1]])
# plt.clabel(contours, inline=True, fontsize=8, colors='white')

# plt.show()

class BlackScholesOptionPricingModel():
    def __init__(self, K, S0, T, r, sigma, opttype='C'):
        self.asset_price = S0
        self.strike_price = K
        self.time_to_maturity = T
        self.asset_volatility = sigma
        self.risk_free_rate = r
        self.opttype = opttype


    def calculate_derivative_price(self):
        d1 = (np.log(self.asset_price / self.strike_price) + (self.risk_free_rate + ((self.asset_volatility ** 2))/2)*self.time_to_maturity) / self.asset_volatility * np.sqrt(self.time_to_maturity)
        d2 = (np.log(self.asset_price / self.strike_price) + (self.risk_free_rate - ((self.asset_volatility ** 2))/2)*self.time_to_maturity) / self.asset_volatility * np.sqrt(self.time_to_maturity)
        return (self.asset_price * norm.cdf(d1)) - (self.strike_price*(np.exp(-1*self.risk_free_rate*self.time_to_maturity)*norm.cdf(d2)))

bsm = BlackScholesOptionPricingModel(100, 100, 1, 0.05, 0.2, 'C')
print(bsm.calculate_derivative_price())