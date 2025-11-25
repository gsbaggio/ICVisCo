from numpy import *
from scipy import signal
import argparse
from PIL import Image
import os

def BPP(data):
    data = data.reshape(-1)
    zero = sum(isclose(around(data), 0))
    sizeOriginal = len(data)
    sizeCompressed = sizeOriginal - zero
    return 8*sizeCompressed/sizeOriginal  


def WSSSIM(img1, img2, K1 = .01, K2 = .03, L = 255):
    def __fspecial_gauss(size, sigma):
        x, y = mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    def __weights(height, width):
        deltaTheta = 2*pi/width 
        column = asarray([cos( deltaTheta * (j - height/2.+0.5)) for j in range(height)])
        return repeat(column[:, newaxis], width, 1)

    img1 = float64(img1)
    img2 = float64(img2)

    k = 11
    sigma = 1.5
    window = __fspecial_gauss(k, sigma)
    window2 = zeros_like(window); window2[k//2,k//2] = 1 
 
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    W = __weights(*img1.shape)
    Wi = signal.convolve2d(W, window2, 'valid')

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)) * Wi
    mssim = sum(ssim_map)/sum(Wi)

    return mssim


def WSPSNR(img1, img2, max = 255.): # img1 e img2 devem ter shape hx2h e ser em grayscale; max eh o maximo valor possivel em img1 e img2 (verificar se deve ser 1 ou 255)
   def __weights(height, width):
      phis = arange(height+1)*pi/height
      deltaTheta = 2*pi/width 
      column = asarray([deltaTheta * (-cos(phis[j+1])+cos(phis[j])) for j in range(height)])
      return repeat(column[:, newaxis], width, 1)

   w = __weights(*img1.shape)
   # from matplotlib import pyplot as plt; plt.imshow(w); plt.show()
   wmse = sum((img1-img2)**2*w)/(4*pi) # É ASSIM MESMO
   return 10*log10(max**2/wmse)

# ----------------- novo

def weightsantigo(height, width):
    phis = arange(height+1)*pi/height
    deltaTheta = 2*pi/width 
    column = asarray([deltaTheta * (-cos(phis[j+1])+cos(phis[j])) for j in range(height)])
    return repeat(column[:, newaxis], width, 1)

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

def WSSSIM_RGB(img1, img2, K1=.01, K2=.03, L=255):
    def __fspecial_gauss(size, sigma):
        x, y = mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    img1 = float64(img1)
    img2 = float64(img2)
    
    k = 11
    sigma = 1.5
    window = __fspecial_gauss(k, sigma)
    window2 = zeros_like(window)
    window2[k//2, k//2] = 1 
 
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    
    height, width = img1.shape[0], img1.shape[1]
    W = weights(height, width)
    Wi = signal.convolve2d(W, window2, 'valid')
    
    weight_sum = sum(Wi)
    
    wsssim_channels = zeros(3)
    
    for c in range(3):
        channel1 = img1[:, :, c]
        channel2 = img2[:, :, c]
        
        mu1 = signal.convolve2d(channel1, window, 'valid')
        mu2 = signal.convolve2d(channel2, window, 'valid')
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = signal.convolve2d(channel1 * channel1, window, 'valid') - mu1_sq
        sigma2_sq = signal.convolve2d(channel2 * channel2, window, 'valid') - mu2_sq
        sigma12 = signal.convolve2d(channel1 * channel2, window, 'valid') - mu1_mu2
        
        numerator = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = (numerator / denominator) * Wi
        
        wsssim_channels[c] = sum(ssim_map) / weight_sum
    
    return mean(wsssim_channels)

def load_image(image_path):
    """
    Carrega uma imagem PNG e retorna como array numpy.
    
    Args:
        image_path: Caminho para a imagem
    
    Returns:
        Array numpy com shape (height, width, 3) e valores em [0, 255]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    img = Image.open(image_path)
    
    # Converte para RGB se necessário (remove canal alpha se existir)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        raise ValueError(f"Formato de imagem não suportado: {img.mode}. Use RGB ou RGBA.")
    
    return asarray(img)


def main():
    """
    Função principal que processa argumentos de linha de comando e calcula métricas.
    """
    parser = argparse.ArgumentParser(
        description='Calcula métricas de distorção Weighted Spherical (WS-MSE, WS-PSNR, WS-SSIM) entre duas imagens 360 graus.'
    )
    parser.add_argument('--original', '-o', required=True, 
                        help='Nome do arquivo da imagem original (ex: original.png)')
    parser.add_argument('--fake', '-f', required=True,
                        help='Nome do arquivo da imagem reconstruída/fake (ex: fake.png)')
    
    args = parser.parse_args()
    
    # Obtém o diretório do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Constrói caminhos completos
    original_path = os.path.join(script_dir, args.original)
    fake_path = os.path.join(script_dir, args.fake)
    
    print("=" * 70)
    print("Calculando métricas de distorção Weighted Spherical para imagens 360°")
    print("=" * 70)
    print(f"\nImagem original: {args.original}")
    print(f"Imagem fake:     {args.fake}\n")
    
    try:
        # Carrega as imagens
        print("Carregando imagens...")
        img_original = load_image(original_path)
        img_fake = load_image(fake_path)
        
        # Verifica se as dimensões são compatíveis
        if img_original.shape != img_fake.shape:
            raise ValueError(
                f"As imagens têm dimensões diferentes!\n"
                f"Original: {img_original.shape}, Fake: {img_fake.shape}"
            )
        
        print(f"Dimensões das imagens: {img_original.shape[0]} x {img_original.shape[1]} x {img_original.shape[2]}\n")
        
        # Calcula as métricas
        print("Calculando WS-MSE...")
        wsmse = WSMSE_RGB(img_original, img_fake)
        
        print("Calculando WS-PSNR...")
        wspsnr = WSPSNR_RGB(img_original, img_fake)
        
        print("Calculando WS-SSIM...")
        wsssim = WSSSIM_RGB(img_original, img_fake)
        
        # Exibe os resultados
        print("\n" + "=" * 70)
        print("RESULTADOS:")
        print("=" * 70)
        print(f"WS-MSE:   {wsmse:.6f}")
        print(f"WS-PSNR:  {wspsnr:.4f} dB")
        print(f"WS-SSIM:  {wsssim:.6f}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\nErro: {e}")
        print(f"Certifique-se de que os arquivos estão no diretório: {script_dir}")
        return 1
    except Exception as e:
        print(f"\nErro ao processar imagens: {e}")
        return 1
    
    # print(weightsnovo(5,8))
    # print(weights(5,8))
    
    return 0


if __name__ == "__main__":
    exit(main())