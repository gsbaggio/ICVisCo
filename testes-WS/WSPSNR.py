from numpy import *

def weights(height, width): # calculo da matriz de pesos, otimizada
    phis = arange(height+1)*pi/height
    deltaTheta = 2*pi/width
    column = deltaTheta * (-cos(phis[1:]) + cos(phis[:-1]))
    return repeat(column[:, newaxis], width, 1)

def WSPSNR_RGB(img1, img2, max_val=255.): # cálculo em 3 canais, otimizada
    img1 = float64(img1)
    img2 = float64(img2)
    
    height, width = img1.shape[0], img1.shape[1]

    # calcula os pesos e expande os pesos para shape (height, width, 1)
    w = weights(height, width)
    w_expanded = w[:, :, newaxis] # (height, width, 1)
    
    # calcula o WS-MSE para todos os canais (olhar o código WSMSE.py)
    squared_diff = (img1 - img2) ** 2
    weighted_squared_diff = squared_diff * w_expanded
    wmse_three_channel = sum(sum(weighted_squared_diff, 0), 0) / (4 * pi)

    # calcula PSNR para cada canal
    wmse_three_channel = where(wmse_three_channel == 0, 1e-10, wmse_three_channel) # evita divisão por zero, pois iria para infinito (ainda fica com um valor muito alto)
    wspsnr_three_channel = 10 * log10(max_val**2 / wmse_three_channel)

    return mean(wspsnr_three_channel)