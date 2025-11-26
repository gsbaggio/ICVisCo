from numpy import *

def weights(height, width): # calculo da matriz de pesos, otimizada
    phis = arange(height+1)*pi/height
    deltaTheta = 2*pi/width
    column = deltaTheta * (-cos(phis[1:]) + cos(phis[:-1]))
    return repeat(column[:, newaxis], width, 1)/(4*pi)

def teste_weights(height, width):
    phis = arange(height+1)*pi/height
    print(phis)
    deltaTheta = 2*pi/width
    print(deltaTheta)
    column = deltaTheta * (-cos(phis[1:]) + cos(phis[:-1]))
    return repeat(column[:, newaxis], width, 1)

teste = weights(128,256)
print(sum(teste))

teste = teste_weights(128,256)
print(sum(teste))