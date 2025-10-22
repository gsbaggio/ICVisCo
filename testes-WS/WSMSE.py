from numpy import *

def weights(height, width): # calculo da matriz de pesos, otimizada
    phis = arange(height+1)*pi/height
    deltaTheta = 2*pi/width
    column = deltaTheta * (-cos(phis[1:]) + cos(phis[:-1]))
    return repeat(column[:, newaxis], width, 1)

def WSMSE_RGB(img1, img2): # cálculo em 3 canais, otimizada
    img1 = float64(img1)
    img2 = float64(img2)
    
    height, width = img1.shape[0], img1.shape[1]
    
    # calcula os pesos e expande os pesos para shape (height, width, 1)
    w = weights(height, width)
    w_expanded = w[:, :, newaxis] # (height, width, 1)
    
    # (img1 - img2)^2 tem shape (height, width, 3)
    # w_expanded tem shape (height, width, 1)
    # a multiplicação faz broadcast automaticamente
    squared_diff = (img1 - img2) ** 2
    weighted_squared_diff = squared_diff * w_expanded
    
    # soma sobre altura e largura, mantendo os canais separados
    r = 1
    wmse_three_channel = sum(sum(weighted_squared_diff, 0), 0) / (4 * pi * r)

    # média dos 3 canais
    return mean(wmse_three_channel)