"""
Script para validar se WSPSNR_RGB e WSSSIM_RGB estão corretos.

Estratégia:
1. Criar imagens grayscale sintéticas
2. Criar versões RGB dessas imagens com os 3 canais idênticos
3. Comparar os resultados das métricas grayscale vs RGB
4. Se estiver correto, os valores devem ser idênticos
"""

from numpy import *
from scipy import signal
from PIL import Image
import os

def WSPSNR(img1, img2, max_val=255.):
    def __weights(height, width):
        phis = arange(height+1)*pi/height
        deltaTheta = 2*pi/width 
        column = asarray([deltaTheta * (-cos(phis[j+1])+cos(phis[j])) for j in range(height)])
        return repeat(column[:, newaxis], width, 1)

    w = __weights(*img1.shape)
    wmse = sum((img1-img2)**2*w)/(4*pi)
    return 10*log10(max_val**2/wmse)


def WSSSIM(img1, img2, K1=.01, K2=.03, L=255):
    def __fspecial_gauss(size, sigma):
        x, y = mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    def __weights(height, width):
        deltaTheta = 2*pi/width 
        column = asarray([cos(deltaTheta * (j - height/2.+0.5)) for j in range(height)])
        return repeat(column[:, newaxis], width, 1)

    img1 = float64(img1)
    img2 = float64(img2)

    k = 11
    sigma = 1.5
    window = __fspecial_gauss(k, sigma)
    window2 = zeros_like(window)
    window2[k//2,k//2] = 1 
 
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

def weights(height, width):
    """Cálculo otimizado da matriz de pesos"""
    phis = arange(height+1)*pi/height
    deltaTheta = 2*pi/width
    column = deltaTheta * (-cos(phis[1:]) + cos(phis[:-1]))
    return repeat(column[:, newaxis], width, 1)


def WSPSNR_RGB(img1, img2, max_val=255.):
    img1 = float64(img1)
    img2 = float64(img2)
    
    height, width = img1.shape[0], img1.shape[1]

    w = weights(height, width)
    w_expanded = w[:, :, newaxis]
    
    squared_diff = (img1 - img2) ** 2
    weighted_squared_diff = squared_diff * w_expanded
    wmse_three_channel = sum(sum(weighted_squared_diff, 0), 0) / (4 * pi)

    wmse_three_channel = where(wmse_three_channel == 0, 1e-10, wmse_three_channel)
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


def create_test_images():
    """
    Cria imagens de teste grayscale sintéticas com formato equiretangular.
    Retorna: (img_gray_original, img_gray_fake)
    """
    # Criar imagens 100x200 (formato equiretangular típico: height x 2*height)
    height, width = 100, 200
    
    # Imagem original: gradiente + padrão
    img_original = zeros((height, width))
    for i in range(height):
        for j in range(width):
            img_original[i, j] = (i * 2.55) % 256  # Gradiente vertical
    
    # Adicionar algum ruído/diferença para a imagem fake
    img_fake = img_original.copy()
    random.seed(42)
    noise = random.randint(-10, 10, (height, width))
    img_fake = clip(img_fake + noise, 0, 255)
    
    return img_original.astype(uint8), img_fake.astype(uint8)


def gray_to_rgb(img_gray):
    """
    Converte imagem grayscale para RGB com 3 canais idênticos.
    """
    return stack([img_gray, img_gray, img_gray], axis=2)


def save_images(img_gray_original, img_gray_fake):
    """
    Salva as imagens de teste.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Salvar grayscale
    Image.fromarray(img_gray_original, mode='L').save(
        os.path.join(script_dir, 'test_gray_original.png')
    )
    Image.fromarray(img_gray_fake, mode='L').save(
        os.path.join(script_dir, 'test_gray_fake.png')
    )
    
    # Salvar RGB (3 canais idênticos)
    img_rgb_original = gray_to_rgb(img_gray_original)
    img_rgb_fake = gray_to_rgb(img_gray_fake)
    
    Image.fromarray(img_rgb_original, mode='RGB').save(
        os.path.join(script_dir, 'test_rgb_original.png')
    )
    Image.fromarray(img_rgb_fake, mode='RGB').save(
        os.path.join(script_dir, 'test_rgb_fake.png')
    )
    


def run_validation_tests():
    # Criar imagens
    img_gray_original, img_gray_fake = create_test_images()
    img_rgb_original = gray_to_rgb(img_gray_original)
    img_rgb_fake = gray_to_rgb(img_gray_fake)
    
    # Salvar imagens
    save_images(img_gray_original, img_gray_fake)
    
    print(f"Dimensões grayscale: {img_gray_original.shape}")
    print(f"Dimensões RGB: {img_rgb_original.shape}")
    
    # ==================== TESTE 1: WSPSNR ====================
    print("\n" + "=" * 80)
    print("TESTE 1: WSPSNR")
    print("=" * 80)
    
    wspsnr_gray = WSPSNR(
        float64(img_gray_original), 
        float64(img_gray_fake)
    )
    
    wspsnr_rgb = WSPSNR_RGB(
        float64(img_rgb_original), 
        float64(img_rgb_fake)
    )
    
    print(f"\nWS-PSNR (versão grayscale): {wspsnr_gray:.10f} dB")
    print(f"WS-PSNR (versão RGB):       {wspsnr_rgb:.10f} dB")
    print(f"Diferença absoluta:         {abs(wspsnr_gray - wspsnr_rgb):.10e}")

    # ==================== TESTE 2: WSSSIM ====================
    print("\n" + "=" * 80)
    print("TESTE 2: WSSSIM")
    print("=" * 80)
    
    wsssim_gray = WSSSIM(
        float64(img_gray_original), 
        float64(img_gray_fake)
    )
    
    wsssim_rgb = WSSSIM_RGB(
        float64(img_rgb_original), 
        float64(img_rgb_fake)
    )
    
    print(f"\nWS-SSIM (versão grayscale): {wsssim_gray:.10f}")
    print(f"WS-SSIM (versão RGB):       {wsssim_rgb:.10f}")
    print(f"Diferença absoluta:         {abs(wsssim_gray - wsssim_rgb):.10e}")

if __name__ == "__main__":
    run_validation_tests()
