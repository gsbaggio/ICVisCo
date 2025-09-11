# https://tltsilveira.github.io/public/CSGraduationWork.pdf material do thiago

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

def imprime_matriz(matriz):
    for i in range(len(matriz)):
        print(matriz[i])

def visualizar_matriz(matriz):
    plt.figure(figsize=(8, 8))
    plt.imshow(matriz, cmap='gray', vmin=0, vmax=255)
    plt.title("Grafico")
    plt.colorbar(label='Intensidade')
    plt.show()

def reconstruir_matriz_de_blocos(blocos, altura, largura):
    matriz_reconstruida = np.zeros((altura, largura))
    bloco_idx = 0
    
    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            if bloco_idx < len(blocos):
                altura_bloco = min(8, altura - i)
                largura_bloco = min(8, largura - j)
                matriz_reconstruida[i:i+altura_bloco, j:j+largura_bloco] = blocos[bloco_idx][:altura_bloco, :largura_bloco]
                bloco_idx += 1
    
    return matriz_reconstruida

def obter_matriz_exemplo():

    dados_matriz = [
        [127, 123, 125, 120, 126, 123, 127, 128],
        [142, 135, 144, 143, 140, 145, 142, 140],
        [128, 126, 128, 122, 125, 125, 122, 129],
        [132, 144, 144, 139, 140, 149, 140, 142],
        [128, 124, 128, 126, 127, 120, 128, 129],
        [133, 142, 141, 141, 143, 140, 146, 138],
        [124, 127, 128, 129, 121, 128, 129, 128],
        [134, 143, 140, 139, 136, 140, 138, 141]
    ]

    matriz_a8 = np.array(dados_matriz, dtype=np.int16)

    return matriz_a8

def gerar_matriz_aleatoria(altura, largura):
    matriz_aleatoria = np.random.randint(0, 256, size=(altura, largura), dtype=np.uint8)
    return matriz_aleatoria

def criar_blocos_8x8(matriz):
    altura, largura = matriz.shape
    blocos = []
    
    for i in range(0, altura, 8):
        for j in range(0, largura, 8):
            bloco = matriz[i:i+8, j:j+8]
            blocos.append(bloco)

    return blocos

def cria_matriz_transformacao_DCT():
    N = 8
    matriz_DCT = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u == 0:
                au = np.sqrt(1 / N)
            else:
                au = np.sqrt(2 / N)
            
            matriz_DCT[u, v] = au * np.cos((2 * v + 1) * u * np.pi / (2 * N))
    return matriz_DCT

def DCT_matricial(blocos):
    N = 8
    matriz_DCT = cria_matriz_transformacao_DCT()
    blocos_dct = []
    for bloco in blocos:
        bloco_dct = np.dot(np.dot(matriz_DCT, bloco), matriz_DCT.T)
        blocos_dct.append(bloco_dct)
    return blocos_dct

def cria_matriz_quantizacao(R):
    matriz = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            matriz[i, j] = 1 + (i + j) * R
    return matriz

def quantizacao(bloco_dct, matriz_quantizacao):
    bloco_quantizado = matriz_quantizacao * np.round(bloco_dct / matriz_quantizacao)
    return bloco_quantizado

def quantizacao_blocos(blocos_dct, R):
    matriz_quantizacao = cria_matriz_quantizacao(R)
    blocos_quantizados = []
    for bloco_dct in blocos_dct:
        bloco_quantizado = quantizacao(bloco_dct, matriz_quantizacao)
        blocos_quantizados.append(bloco_quantizado)
    return blocos_quantizados

def DCT_inversa_matricial(blocos_dct):
    N = 8
    matriz_DCT = cria_matriz_transformacao_DCT()
    blocos_reconstruidos = []
    for bloco_dct in blocos_dct:
        bloco_reconstruido = np.round(np.dot(np.dot(matriz_DCT.T, bloco_dct), matriz_DCT))
        blocos_reconstruidos.append(bloco_reconstruido)
    return blocos_reconstruidos

def calcular_PSNR(matriz_original, matriz_reconstruida):
    psnr_value = peak_signal_noise_ratio(matriz_original, matriz_reconstruida, data_range=255)
    return psnr_value

def porcentagem_zeros(blocos_quantizados):
    total_elementos = 0
    zeros = 0
    for bloco in blocos_quantizados:
        total_elementos += bloco.size
        zeros += np.count_nonzero(bloco == 0)
    return (zeros / total_elementos) * 100

def plot_rd_curve(matriz_original, valores_R):
    porcentagens = []
    psnrs = []
    for R in valores_R:
        blocos = criar_blocos_8x8(matriz_original)
        blocos_dct = DCT_matricial(blocos)
        blocos_quant = quantizacao_blocos(blocos_dct, R)
        blocos_rec = DCT_inversa_matricial(blocos_quant)
        matriz_rec = reconstruir_matriz_de_blocos(blocos_rec, matriz_original.shape[0], matriz_original.shape[1])
        porcentagens.append(porcentagem_zeros(blocos_quant))
        psnrs.append(calcular_PSNR(matriz_original, matriz_rec))
    plt.figure(figsize=(10, 6))
    plt.plot(psnrs, porcentagens, 'o-')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Porcentagem de Zeros (%)')
    plt.title('RD-Curve')
    plt.grid(True)
    plt.show()

ALTURA_IMG = 8
LARGURA_IMG = 8
valores_R = [2, 5, 7, 10]

plot_rd_curve(obter_matriz_exemplo(), valores_R)

# matriz_grad = obter_matriz_exemplo()

# matriz_grad = gerar_matriz_aleatoria(ALTURA_IMG, LARGURA_IMG)

# print(matriz_grad)

# visualizar_matriz(matriz_grad)

# blocos = criar_blocos_8x8(matriz_grad)

# blocos_dct_matricial = DCT_matricial(blocos)

# blocos_quantizados = quantizacao_blocos(blocos_dct_matricial, R)

# print(blocos_quantizados[0])

# blocos_reconstruidos = DCT_inversa_matricial(blocos_quantizados)

# matriz_final = reconstruir_matriz_de_blocos(blocos_reconstruidos, ALTURA_IMG, LARGURA_IMG)

# print(matriz_final)

# visualizar_matriz(matriz_final)

# print("PSNR:", calcular_PSNR(matriz_grad, matriz_final))