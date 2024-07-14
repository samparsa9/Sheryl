# The stock tree will be represented using nodes (i, j) and initial stock price of S0
# Si,j = S0u^(j)d^(i-j)
# Ci,j = contract price at each node in the tree (i,j), where CNj = final payoff function that we will define
# Going to price European Call option, so CNj = max(SNj - K,0)

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.stats import norm
# Initializing parameters
S0 = 100        # Current stock price
K = 100         # Strike price of the option
T = 1           # Time to maturity in years
sigma = 0.2     # Volatility of stock (annualized standard deviation of the stock's returns)
r = 0.06        # Annual risk-free rate
N = 1000        # Number of time steps

option_type = 'C' # C to represent a call, P to represent a put

def binomial_option_pricing_model(S0, K, T, sigma, r, N, opttype='C'):
    # At each step in the tree, the price can move up or down by a factor of u or d respectively
    delta_t = T / N
    u = np.exp(sigma * np.sqrt(delta_t))
    d = np.exp(-1 * sigma * np.sqrt(delta_t))
    # Calculating risk neutral probability
    P = ((np.exp(r*delta_t) - d)) / (u - d)

    stock_price_tree = []
    derivative_price_tree = []
    for level_in_tree in range(1,N+2):
        stock_price_tree.append([0 for _ in range(0, level_in_tree)])
        derivative_price_tree.append([0 for _ in range(0, level_in_tree)])
    
    # First need to calculate the initial values for the last layer of the tree

    for i in range(N+1): # Tree looks like this [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
        # print(f"Our binomial tree is: {stock_price_tree}")
        # print(f"Length of the list nodes at index {i}: {len(stock_price_tree[i])}")
        for j in range(0, len(stock_price_tree[i])):
            # print(f"Our binomial tree is: {stock_price_tree}")
            calculation = S0 * (u ** j) * (d ** i)
            stock_price_tree[i][j] = S0 * (u ** j) * (d ** (i-j))

    for k in range(0, N+1): # Tree looks like this [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
        if opttype == 'C':
            derivative_price_tree[-1][k] = max(stock_price_tree[-1][k] - K, 0) # For call
        elif opttype == 'P':
            derivative_price_tree[-1][k] = max(K - stock_price_tree[-1][k], 0) # For Put


    for x in range(N-1, -1, -1):
        for y in range(len(derivative_price_tree[x])):
            derivative_price_tree[x][y] = np.exp(-1*r*delta_t) * (P * derivative_price_tree[x+1][y+1] + (1-P)*derivative_price_tree[x+1][y])
    # derivative_price_tree[k][n] = np.exp(-1*r*delta_t) * P * stock_price_tree[k-1][n-1] + (1-P)*derivative_price_tree[k-1][n]

    #stock_price_tree[i][j] = np.exp(-1*r*delta_t) * P * stock_price_tree[i-1][j-1] + (1-P)*stock_price_tree[i-1][j]


    return derivative_price_tree[0][0]

# print(calculate_derivative_price(S0, K, T, sigma, r, N))

volatility_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
r_list =          [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

option_prices = np.zeros((len(volatility_list), len(r_list)))

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

def black_scholes_model(K, S0, T, r, sigma, opttype='C'):
    d1 = (np.log(S0 / K) + (r + ((sigma ** 2))/2)*T) / sigma * np.sqrt(T)
    d2 = (np.log(S0 / K) + (r - ((sigma ** 2))/2)*T) / sigma * np.sqrt(T)
    return (S0 * norm.cdf(d1)) - (K*(np.exp(-1*r*T)*norm.cdf(d2)))

print(black_scholes_model(100, 100, 1, 0.05, 0.20))