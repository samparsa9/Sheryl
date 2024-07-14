import numpy as np
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

print(binomial_option_pricing_model(100, 100, 1, 0.2, 0.05, 5, 'C'))