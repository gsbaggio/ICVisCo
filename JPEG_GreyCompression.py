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

def transformada_DCT_blocos(blocos):
    blocos_dct = []
    for bloco in blocos:
        bloco_dct = DCT_2D_metodo2(bloco)
        blocos_dct.append(bloco_dct)
    return blocos_dct

def quantizacao(bloco_dct, matriz_quantizacao):
    bloco_quantizado = np.round(bloco_dct / matriz_quantizacao)
    return bloco_quantizado

def quantizacao_blocos(blocos_dct, matriz_quantizacao):
    blocos_quantizados = []
    for bloco_dct in blocos_dct:
        bloco_quantizado = quantizacao(bloco_dct, matriz_quantizacao)
        blocos_quantizados.append(bloco_quantizado)
    return blocos_quantizados

def zig_zag_scan(blocos_quantizados):
    sequencias = []
    for bloco in blocos_quantizados:
        sequencia = []
        for s in range(15):
            if s % 2 == 0:
                for i in range(s + 1):
                    j = s - i
                    if i < 8 and j < 8:
                        sequencia.append(bloco[i, j])
            else:
                for i in range(s + 1):
                    j = s - i
                    if j < 8 and i < 8:
                        sequencia.append(bloco[j, i])
        sequencias.append(sequencia)
    return sequencias   

def

ALTURA_IMG = 128
LARGURA_IMG = 128

matriz = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]

C = [
    [0, 1, 5, 6, 14, 15, 27, 28],
    [2, 4, 7, 13, 16, 26, 29, 42],
    [3, 8, 12, 17, 25, 30, 41, 43],
    [9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]
]

matriz_grad = criar_matriz_gradiente(ALTURA_IMG, LARGURA_IMG)

print(matriz_grad)

matriz_grad = level_shift(matriz_grad)

print(matriz_grad)

blocos = criar_blocos_8x8(matriz_grad)

print(blocos)

blocos_dct = transformada_DCT_blocos(blocos)

print(blocos_dct)

blocos_quantizados = quantizacao_blocos(blocos_dct, np.array(matriz))

print(blocos_quantizados)