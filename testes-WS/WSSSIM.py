from numpy import *
from scipy import signal
from PIL import Image

def weights(height, width): # calculo da matriz de pesos, otimizada
    phis = arange(height+1)*pi/height
    deltaTheta = 2*pi/width
    column = deltaTheta * (-cos(phis[1:]) + cos(phis[:-1]))
    return repeat(column[:, newaxis], width, 1)

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

img1 = asarray(Image.open('original.png').convert('RGB')) # shape precisa ser (altura, largura, 3) !!!
img2 = asarray(Image.open('fake.png').convert('RGB')) # shape precisa ser (altura, largura, 3) !!!

print(WSSSIM_RGB(img1, img2))