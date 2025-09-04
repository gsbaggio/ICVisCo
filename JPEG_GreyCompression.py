# https://tltsilveira.github.io/public/CSGraduationWork.pdf material do thiago

import numpy as np

def criar_matriz_gradiente(altura, largura):
    linha_gradiente = np.linspace(0, 255, largura)
    
    matriz = np.tile(linha_gradiente, (altura, 1))

    return matriz

def level_shift(matriz):
    matriz = matriz - 128
    matriz = matriz.astype(np.int8)
    return matriz

def criar_blocos_8x8(matriz):
    altura, largura = matriz.shape
    blocos = []
    
    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco = matriz[i:i+8, j:j+8]
            blocos.append(bloco)

    return blocos

def DCT_2D_metodo2(bloco): # como esta no artigo do Thiago
    N = 8
    bloco_dct = np.zeros((N, N))

    for u in range(N):
        for v in range(N):
            if u == 0:
                Cu = 1 / np.sqrt(N)
            else:
                Cu = np.sqrt(2 / N)
            if v == 0:
                Cv = 1 / np.sqrt(N)
            else:
                Cv = np.sqrt(2 / N)

            sum_val = 0
            for x in range(N):
                for y in range(N):
                    sum_val += bloco[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))

            bloco_dct[u, v] = Cu * Cv * sum_val

    return bloco_dct

def DCT_2D(bloco): # outro jeito que quero testar
    N = 8
    bloco_dct = np.zeros((N, N))

    for u in range(N):
        for v in range(N):
            if u == 0:
                Cu = 1 / np.sqrt(2)
            else:
                Cu = 1

            if v == 0:
                Cv = 1 / np.sqrt(2)
            else:
                Cv = 1

            sum_val = 0
            for x in range(N):
                for y in range(N):
                    sum_val += bloco[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
            bloco_dct[u, v] = Cu * Cv * sum_val / 4

    return bloco_dct


ALTURA_IMG = 128
LARGURA_IMG = 128

matriz_grad = criar_matriz_gradiente(ALTURA_IMG, LARGURA_IMG)

print(matriz_grad)

matriz_grad = level_shift(matriz_grad)

print(matriz_grad)

blocos = criar_blocos_8x8(matriz_grad)

print(blocos)