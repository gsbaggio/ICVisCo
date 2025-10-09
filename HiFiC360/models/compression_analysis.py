#!/usr/bin/env python3
"""
Análise de compressão de imagens com múltiplas métricas de distorção.
Plota gráficos scatter de BPP vs métricas de distorção para diferentes métodos de compressão.
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# Imports opcionais - permite rodar sem TensorFlow/HiFiC instalados
try:
    import tensorflow.compat.v1 as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Aviso: TensorFlow não disponível. Algumas métricas podem não funcionar.")

try:
    from hific.evaluate import get_psnr
    HIFIC_AVAILABLE = True
except ImportError:
    HIFIC_AVAILABLE = False
    print("Aviso: Módulo HiFiC não disponível. Usando implementação local do PSNR.")

try:
    from hific.model import LPIPSLoss
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Aviso: LPIPS não disponível.")


def calculate_bpp_from_files(original_path, compressed_path):
    """
    Calcula BPP (bits per pixel) comparando tamanho do arquivo original PNG 
    com o arquivo comprimido .tfci
    
    Args:
        original_path: caminho para imagem original (.png)
        compressed_path: caminho para arquivo comprimido (.tfci)
    
    Returns:
        bpp: bits per pixel
    """
    # Tamanho do arquivo comprimido em bytes
    compressed_size = os.path.getsize(compressed_path)
    
    # Carrega imagem original para obter dimensões
    img = Image.open(original_path)
    width, height = img.size
    
    # Calcula BPP
    total_pixels = width * height
    bpp = (compressed_size * 8) / total_pixels
    
    return bpp


def calculate_psnr(original_path, decompressed_path):
    """
    Calcula PSNR entre imagem original e decomprimida.
    
    Args:
        original_path: caminho para imagem original
        decompressed_path: caminho para imagem decomprimida
    
    Returns:
        psnr: Peak Signal-to-Noise Ratio
    """
    # Carrega imagens
    original_img = Image.open(original_path)
    decompressed_img = Image.open(decompressed_path)
    
    # Converte RGBA para RGB se necessário
    if original_img.mode == 'RGBA':
        original_img = original_img.convert('RGB')
    if decompressed_img.mode == 'RGBA':
        decompressed_img = decompressed_img.convert('RGB')
    
    original = np.array(original_img)
    decompressed = np.array(decompressed_img)
    
    # Garante que as imagens tenham as mesmas dimensões
    if original.shape != decompressed.shape:
        raise ValueError(f"Dimensões diferentes: {original.shape} vs {decompressed.shape}")
    
    # Usa função do HiFiC se disponível, senão implementação local
    if HIFIC_AVAILABLE:
        return get_psnr(original, decompressed)
    else:
        # Implementação local do PSNR
        mse = np.mean(np.square(original.astype(np.float32) - decompressed.astype(np.float32)))
        if mse == 0:
            return float('inf')  # Imagens idênticas
        psnr = 20. * np.log10(255.) - 10. * np.log10(mse)
        return psnr


class SimpleLPIPSLoss:
    """LPIPS Loss implementação simplificada para TF 1.x."""
    
    def __init__(self, weight_path):
        # Verifica se o arquivo de pesos existe
        if not os.path.exists(weight_path):
            from hific import helpers
            helpers.ensure_lpips_weights_exist(weight_path)
        
        # Carrega o grafo LPIPS
        self.graph_def = tf.GraphDef()
        with open(weight_path, "rb") as f:
            self.graph_def.ParseFromString(f.read())
    
    def __call__(self, fake_image, real_image):
        """
        Calcula LPIPS assumindo inputs em [0, 1].
        """
        # Move inputs para [-1, 1] e formato NCHW
        def _transpose_to_nchw(x):
            return tf.transpose(x, (0, 3, 1, 2))
        
        fake_image_nchw = _transpose_to_nchw(fake_image * 2 - 1.0)
        real_image_nchw = _transpose_to_nchw(real_image * 2 - 1.0)
        
        # Importa o grafo LPIPS
        loss = tf.import_graph_def(
            self.graph_def,
            input_map={"0:0": fake_image_nchw, "1:0": real_image_nchw},
            return_elements=["Reshape_10:0"]
        )[0]
        
        return tf.reduce_mean(loss)  # Loss é N111, toma média para obter escalar


class LPIPSCalculator:
    """Classe para calcular LPIPS usando implementação simplificada com suporte a GPU."""
    
    def __init__(self, weight_path="lpips_weight__net-lin_alex_v0.1.pb", use_gpu=True):
        self.weight_path = weight_path
        self.use_gpu = use_gpu
        self.lpips_loss = None
        self.session = None
        self._initialized = False
    
    def _initialize(self):
        """Inicializa o modelo LPIPS e a sessão TensorFlow."""
        if self._initialized:
            return True
            
        try:
            # Força CPU se use_gpu=False, ou detecta GPU se use_gpu=True
            if not self.use_gpu:
                print("Forçando uso de CPU...")
                # Desabilita completamente a GPU
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                # Configuração para CPU apenas
                config = tf.ConfigProto()
                config.allow_soft_placement = True
                config.log_device_placement = False
                actual_use_gpu = False
            else:
                gpu_available = TF_AVAILABLE and tf.test.is_gpu_available()
                actual_use_gpu = gpu_available
                
                if actual_use_gpu:
                    print("Inicializando LPIPS com GPU...")
                    # Configuração otimizada para GPU
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    config.gpu_options.per_process_gpu_memory_fraction = 0.7
                    config.allow_soft_placement = True
                else:
                    print("GPU não disponível, usando CPU...")
                    # Configuração para CPU
                    config = tf.ConfigProto()
                    config.allow_soft_placement = True
                    config.log_device_placement = False
            
            # Usa a implementação simplificada importada
            from simple_lpips import SimpleLPIPSLoss
            self.lpips_loss = SimpleLPIPSLoss(self.weight_path, use_gpu=actual_use_gpu)
            
            # Cria placeholders para as imagens (formato esperado pelo LPIPS: [0, 1])
            self.real_image_ph = tf.placeholder(tf.float32, [None, None, None, 3], name='real_image')
            self.fake_image_ph = tf.placeholder(tf.float32, [None, None, None, 3], name='fake_image')
            
            # Calcula o loss LPIPS
            self.lpips_value = self.lpips_loss(self.fake_image_ph, self.real_image_ph)
            
            # Cria sessão com configuração apropriada
            self.session = tf.Session(config=config)
            
            self._initialized = True
            self.use_gpu = actual_use_gpu  # Atualiza o estado real
            device_type = "GPU" if actual_use_gpu else "CPU"
            print(f"LPIPS inicializado com sucesso ({device_type})")
            return True
            
        except Exception as e:
            print(f"Erro inicializando LPIPS: {e}")
            # Tenta fallback para CPU se falhou com GPU
            if self.use_gpu:
                print("Tentando fallback para CPU...")
                self.use_gpu = False
                tf.reset_default_graph()
                return self._initialize()
            self._initialized = False
            return False
    
    def calculate(self, original_image, decompressed_image):
        """
        Calcula LPIPS entre duas imagens.
        
        Args:
            original_image: numpy array da imagem original (H, W, 3) em [0, 255]
            decompressed_image: numpy array da imagem decomprimida (H, W, 3) em [0, 255]
        
        Returns:
            lpips_score: float com o valor LPIPS
        """
        if not self._initialize():
            return None
        
        try:
            # Normaliza imagens para [0, 1] como esperado pelo LPIPS
            real_img = original_image.astype(np.float32) / 255.0
            fake_img = decompressed_image.astype(np.float32) / 255.0
            
            # Adiciona dimensão batch
            real_img = np.expand_dims(real_img, axis=0)
            fake_img = np.expand_dims(fake_img, axis=0)
            
            # Calcula LPIPS
            lpips_score = self.session.run(self.lpips_value, {
                self.real_image_ph: real_img,
                self.fake_image_ph: fake_img
            })
            
            return float(lpips_score)
            
        except Exception as e:
            print(f"Erro calculando LPIPS: {e}")
            return None
    
    def close(self):
        """Fecha a sessão TensorFlow."""
        if self.session:
            self.session.close()
            self.session = None
        tf.reset_default_graph()
        self._initialized = False


class LPIPS360Calculator:
    """Classe para calcular LPIPS 360 com pesos por latitude."""
    
    def __init__(self, weight_path="lpips_weight__net-lin_alex_v0.1.pb", use_gpu=True, 
                 latitude_weight_type='cosine', pole_weight=0.5):
        self.weight_path = weight_path
        self.use_gpu = use_gpu
        self.latitude_weight_type = latitude_weight_type
        self.pole_weight = pole_weight
        self.lpips360_loss = None
        self.session = None
        self._initialized = False
    
    def _initialize(self):
        """Inicializa o modelo LPIPS 360 e a sessão TensorFlow."""
        if self._initialized:
            return True
            
        try:
            # Força CPU se use_gpu=False, ou detecta GPU se use_gpu=True
            if not self.use_gpu:
                print("Forçando uso de CPU para LPIPS 360...")
                # Desabilita completamente a GPU
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                # Configuração para CPU apenas
                config = tf.ConfigProto()
                config.allow_soft_placement = True
                config.log_device_placement = False
                actual_use_gpu = False
            else:
                gpu_available = TF_AVAILABLE and tf.test.is_gpu_available()
                actual_use_gpu = gpu_available
                
                if actual_use_gpu:
                    print("Inicializando LPIPS 360 com GPU...")
                    # Configuração otimizada para GPU
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    config.gpu_options.per_process_gpu_memory_fraction = 0.7
                    config.allow_soft_placement = True
                else:
                    print("GPU não disponível, usando CPU para LPIPS 360...")
                    # Configuração para CPU
                    config = tf.ConfigProto()
                    config.allow_soft_placement = True
                    config.log_device_placement = False

            # Importa e inicializa LPIPS 360
            try:
                from simple_lpips360 import SimpleLPIPS360Loss
                
                self.lpips360_loss = SimpleLPIPS360Loss(
                    self.weight_path, 
                    use_gpu=actual_use_gpu,
                    latitude_weight_type=self.latitude_weight_type, 
                    pole_weight=self.pole_weight
                )
                
                # Cria placeholders para as imagens (formato esperado: [0, 1])
                self.real_image_ph = tf.placeholder(tf.float32, [None, None, None, 3], name='real_image_360')
                self.fake_image_ph = tf.placeholder(tf.float32, [None, None, None, 3], name='fake_image_360')
                
                # Calcula o loss LPIPS 360
                self.lpips360_value = self.lpips360_loss(self.fake_image_ph, self.real_image_ph)
                
                # Cria sessão com configuração apropriada
                self.session = tf.Session(config=config)
                
                self._initialized = True
                self.use_gpu = actual_use_gpu
                device_type = "GPU" if actual_use_gpu else "CPU"
                print(f"LPIPS 360 inicializado com sucesso ({device_type}) - Tipo: {self.latitude_weight_type}, Peso polar: {self.pole_weight}")
                return True
                
            except ImportError:
                print("Erro: simple_lpips360.py não encontrado.")
                return False
            
        except Exception as e:
            print(f"Erro inicializando LPIPS 360: {e}")
            # Tenta fallback para CPU se falhou com GPU
            if self.use_gpu:
                print("Tentando fallback para CPU...")
                self.use_gpu = False
                tf.reset_default_graph()
                return self._initialize()
            self._initialized = False
            return False
    
    def calculate(self, original_image, decompressed_image):
        """
        Calcula LPIPS 360 entre duas imagens.
        
        Args:
            original_image: numpy array da imagem original (H, W, 3) em [0, 255]
            decompressed_image: numpy array da imagem decomprimida (H, W, 3) em [0, 255]
        
        Returns:
            lpips360_score: LPIPS 360 distance com pesos por latitude
        """
        if not self._initialize():
            return None
            
        try:
            # Normaliza imagens para [0, 1] como esperado pelo LPIPS 360
            real_img = original_image.astype(np.float32) / 255.0
            fake_img = decompressed_image.astype(np.float32) / 255.0
            
            # Adiciona dimensão batch
            real_img = np.expand_dims(real_img, axis=0)
            fake_img = np.expand_dims(fake_img, axis=0)
            
            # Calcula LPIPS 360
            lpips360_score = self.session.run(self.lpips360_value, {
                self.real_image_ph: real_img,
                self.fake_image_ph: fake_img
            })
            
            return float(lpips360_score)
            
        except Exception as e:
            print(f"Erro calculando LPIPS 360: {e}")
            return None
    
    def close(self):
        """Fecha a sessão TensorFlow."""
        if self.session:
            self.session.close()
            self.session = None
        tf.reset_default_graph()
        self._initialized = False


def calculate_lpips(original_path, decompressed_path, lpips_calculator=None):
    """
    Calcula LPIPS entre imagem original e decomprimida usando a implementação original.
    
    Args:
        original_path: caminho para imagem original
        decompressed_path: caminho para imagem decomprimida
        lpips_calculator: instância de LPIPSCalculator (opcional)
    
    Returns:
        lpips_score: LPIPS distance
    """
    if not LPIPS_AVAILABLE or not TF_AVAILABLE:
        print("Aviso: LPIPS/TensorFlow não disponível. Usando aproximação baseada em MSE.")
        # Aproximação usando MSE
        try:
            original_img = Image.open(original_path)
            decompressed_img = Image.open(decompressed_path)
            
            if original_img.mode == 'RGBA':
                original_img = original_img.convert('RGB')
            if decompressed_img.mode == 'RGBA':
                decompressed_img = decompressed_img.convert('RGB')
            
            original = np.array(original_img).astype(np.float32) / 255.0
            decompressed = np.array(decompressed_img).astype(np.float32) / 255.0
            
            mse = np.mean(np.square(original - decompressed))
            lpips_approx = min(mse * 2.0, 1.0)  # Aproximação baseada em MSE
            return lpips_approx
        except Exception as e:
            print(f"Erro calculando aproximação LPIPS: {e}")
            return None
    
    try:
        # Carrega e processa imagens
        original_img = Image.open(original_path)
        decompressed_img = Image.open(decompressed_path)
        
        if original_img.mode == 'RGBA':
            original_img = original_img.convert('RGB')
        if decompressed_img.mode == 'RGBA':
            decompressed_img = decompressed_img.convert('RGB')
        
        original = np.array(original_img)
        decompressed = np.array(decompressed_img)
        
        # Garante que as imagens tenham as mesmas dimensões
        if original.shape != decompressed.shape:
            raise ValueError(f"Dimensões diferentes: {original.shape} vs {decompressed.shape}")
        
        # Usa o calculador LPIPS se fornecido
        if lpips_calculator is not None:
            return lpips_calculator.calculate(original, decompressed)
        else:
            # Cria calculador temporário
            temp_calculator = LPIPSCalculator()
            result = temp_calculator.calculate(original, decompressed)
            temp_calculator.close()
            return result
            
    except Exception as e:
        print(f"Erro calculando LPIPS: {e}")
        return None
    
    # TODO: Implementar cálculo LPIPS
    # Seria necessário carregar as imagens, normalizar e passar pelo modelo
    return None


def calculate_lpips360(original_path, decompressed_path, lpips360_calculator=None):
    """
    Calcula LPIPS 360 entre imagem original e decomprimida com pesos por latitude.
    
    Args:
        original_path: caminho para imagem original
        decompressed_path: caminho para imagem decomprimida
        lpips360_calculator: instância de LPIPS360Calculator (opcional)
    
    Returns:
        lpips360_score: LPIPS 360 distance com pesos por latitude
    """
    if not TF_AVAILABLE:
        print("Aviso: TensorFlow não disponível. LPIPS 360 não pode ser calculado.")
        return None
        
    try:
        # Carrega imagens
        original_img = Image.open(original_path)
        decompressed_img = Image.open(decompressed_path)
        
        if original_img.mode == 'RGBA':
            original_img = original_img.convert('RGB')
        if decompressed_img.mode == 'RGBA':
            decompressed_img = decompressed_img.convert('RGB')
        
        original = np.array(original_img)
        decompressed = np.array(decompressed_img)
        
        # Garante que as imagens tenham as mesmas dimensões
        if original.shape != decompressed.shape:
            raise ValueError(f"Dimensões diferentes: {original.shape} vs {decompressed.shape}")
        
        # Usa o calculador LPIPS 360 se fornecido
        if lpips360_calculator is not None:
            return lpips360_calculator.calculate(original, decompressed)
        else:
            # Cria calculador temporário com configuração padrão
            temp_calculator = LPIPS360Calculator()
            result = temp_calculator.calculate(original, decompressed)
            temp_calculator.close()
            return result
            
    except Exception as e:
        print(f"Erro calculando LPIPS 360: {e}")
        return None


def analyze_compression_folder(base_path, compression_method, metrics=['psnr'], use_gpu=True, 
                              lpips360_weight_type='cosine', lpips360_pole_weight=0.5):
    """
    Analisa uma pasta de compressão específica (ex: hific-hi, hific-lo, hific-mi).
    
    Args:
        base_path: caminho base para a pasta de compressão
        compression_method: nome do método de compressão
        metrics: lista de métricas a calcular ['psnr', 'lpips']
    
    Returns:
        dict: dicionário com métricas calculadas
    """
    original_dir = os.path.join(base_path, 'original')
    compressed_dir = os.path.join(base_path, 'compressed') 
    decompressed_dir = os.path.join(base_path, 'decompressed')
    
    # Verifica se os diretórios existem
    for dir_path in [original_dir, compressed_dir, decompressed_dir]:
        if not os.path.exists(dir_path):
            print(f"Aviso: Diretório não encontrado: {dir_path}")
            return None
    
    results = {
        'method': compression_method,
        'bpp_values': [],
        'psnr_values': [],
        'lpips_values': [],
        'lpips360_values': [],
        'image_names': []
    }
    
    # Inicializa calculador LPIPS uma vez se necessário
    lpips_calculator = None
    if 'lpips' in metrics and LPIPS_AVAILABLE and TF_AVAILABLE:
        try:
            print(f"Inicializando calculador LPIPS para {compression_method}...")
            # Limpa grafo antes de criar novo calculador
            tf.reset_default_graph()
            lpips_calculator = LPIPSCalculator(use_gpu=use_gpu)
            if not lpips_calculator._initialize():
                print("Falha na inicialização do LPIPS, continuando sem LPIPS")
                lpips_calculator = None
        except Exception as e:
            print(f"Erro inicializando LPIPS: {e}")
            lpips_calculator = None
    
    # Inicializa calculador LPIPS 360 uma vez se necessário
    lpips360_calculator = None
    if 'lpips360' in metrics and TF_AVAILABLE:
        try:
            print(f"Inicializando calculador LPIPS 360 para {compression_method}...")
            # Limpa grafo antes de criar novo calculador
            tf.reset_default_graph()
            lpips360_calculator = LPIPS360Calculator(
                use_gpu=use_gpu,
                latitude_weight_type=lpips360_weight_type,
                pole_weight=lpips360_pole_weight
            )
            if not lpips360_calculator._initialize():
                print("Falha na inicialização do LPIPS 360, continuando sem LPIPS 360")
                lpips360_calculator = None
        except Exception as e:
            print(f"Erro inicializando LPIPS 360: {e}")
            lpips360_calculator = None
    
    # Busca arquivos originais
    original_files = glob.glob(os.path.join(original_dir, '*'))
    original_files = [f for f in original_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Encontrados {len(original_files)} arquivos originais em {original_dir}")
    
    for original_path in original_files:
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        # Caminhos correspondentes
        compressed_path = os.path.join(compressed_dir, f"{base_name}.tfci")
        decompressed_path = os.path.join(decompressed_dir, f"{base_name}.png")
        
        # Verifica se todos os arquivos existem
        if not all(os.path.exists(p) for p in [original_path, compressed_path, decompressed_path]):
            print(f"Aviso: Arquivos incompletos para {base_name}")
            continue
        
        try:
            # Calcula BPP
            bpp = calculate_bpp_from_files(original_path, compressed_path)
            results['bpp_values'].append(bpp)
            results['image_names'].append(base_name)
            
            # Calcula PSNR se solicitado
            psnr = None
            if 'psnr' in metrics:
                psnr = calculate_psnr(original_path, decompressed_path)
                results['psnr_values'].append(psnr)
            
            # Calcula LPIPS se solicitado
            lpips_score = None
            if 'lpips' in metrics:
                lpips_score = calculate_lpips(original_path, decompressed_path, lpips_calculator)
                results['lpips_values'].append(lpips_score if lpips_score is not None else 0)
            
            # Calcula LPIPS 360 se solicitado
            lpips360_score = None
            if 'lpips360' in metrics:
                lpips360_score = calculate_lpips360(original_path, decompressed_path, lpips360_calculator)
                results['lpips360_values'].append(lpips360_score if lpips360_score is not None else 0)
            
            # Log com métricas calculadas
            log_parts = [f"BPP: {bpp:.4f}"]
            if 'psnr' in metrics and psnr is not None:
                log_parts.append(f"PSNR: {psnr:.2f} dB")
            if 'lpips' in metrics and lpips_score is not None:
                log_parts.append(f"LPIPS: {lpips_score:.4f}")
            if 'lpips360' in metrics and lpips360_score is not None:
                log_parts.append(f"LPIPS360: {lpips360_score:.4f}")
            
            print(f"Processado: {base_name} - {', '.join(log_parts)}")
            
        except Exception as e:
            print(f"Erro processando {base_name}: {str(e)}")
            continue
    
    # Fecha calculador LPIPS se foi criado
    if lpips_calculator is not None:
        lpips_calculator.close()
    
    # Fecha calculador LPIPS 360 se foi criado
    if lpips360_calculator is not None:
        lpips360_calculator.close()
    
    return results


def plot_compression_analysis(all_results, metrics=['psnr'], output_dir=None):
    """
    Plota gráficos de análise de compressão.
    
    Args:
        all_results: lista de dicionários com resultados
        metrics: métricas para plotar
        output_dir: diretório para salvar gráficos (opcional)
    """
    # Configuração do matplotlib
    plt.style.use('default')
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Configurações específicas por métrica
        if metric == 'psnr':
            ylabel = 'PSNR (dB)'
            title = 'Rate-Distortion: BPP vs PSNR'
        elif metric == 'lpips':
            ylabel = 'LPIPS Distance'
            title = 'Rate-Distortion: BPP vs LPIPS'
        elif metric == 'lpips360':
            ylabel = 'LPIPS 360 Distance'
            title = 'Rate-Distortion: BPP vs LPIPS 360'
        else:
            ylabel = metric.upper()
            title = f'Rate-Distortion: BPP vs {metric.upper()}'
        
        for i, result in enumerate(all_results):
            if not result or not result['bpp_values']:
                continue
                
            color = colors[i % len(colors)]
            
            bpp_mean = np.mean(result['bpp_values'])
            
            if metric == 'psnr' and result['psnr_values']:
                metric_mean = np.mean(result['psnr_values'])
                metric_std = np.std(result['psnr_values'])
                metric_label = f'{result["method"]} ({metric_mean:.2f} dB)'
            elif metric == 'lpips' and result['lpips_values']:
                metric_mean = np.mean(result['lpips_values'])
                metric_std = np.std(result['lpips_values'])
                metric_label = f'{result["method"]} ({metric_mean:.4f})'
            elif metric == 'lpips360' and result['lpips360_values']:
                metric_mean = np.mean(result['lpips360_values'])
                metric_std = np.std(result['lpips360_values'])
                metric_label = f'{result["method"]} ({metric_mean:.4f})'
            else:
                continue
            
            # Plot pontos individuais - sempre círculos com cores diferentes
            ax.scatter(result['bpp_values'], 
                      result[f'{metric}_values'],
                      c=color, marker='o', alpha=0.6, s=50)
            
            # Plot ponto médio com barra de erro (sem label para evitar duplicação)
            ax.errorbar(bpp_mean, metric_mean, yerr=metric_std, 
                       fmt='o', color=color, markersize=10, 
                       capsize=5, capthick=2, linewidth=2)
            
            # Adiciona ponto para a legenda usando scatter (renderiza melhor os círculos)
            ax.scatter([], [], c=color, marker='o', s=100, label=metric_label)
        
        ax.set_xlabel('BPP (bits per pixel)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'compression_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Análise de compressão de imagens')
    parser.add_argument('--base_dir', default='files',
                       help='Diretório base contendo as pastas de métodos')
    parser.add_argument('--methods', nargs='+', default=['hific'],
                       help='Métodos de compressão a analisar')
    parser.add_argument('--metrics', nargs='+', default=['psnr'],
                       choices=['psnr', 'lpips', 'lpips360'],
                       help='Métricas de distorção a calcular')
    parser.add_argument('--output_dir', default='results',
                       help='Diretório para salvar resultados')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Usa GPU para cálculos LPIPS (se disponível)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Força uso de CPU mesmo se GPU estiver disponível')
    parser.add_argument('--lpips360_weight_type', default='cosine',
                       choices=['cosine', 'linear', 'quadratic'],
                       help='Tipo de peso por latitude para LPIPS 360')
    parser.add_argument('--lpips360_pole_weight', type=float, default=0.5,
                       help='Peso para regiões polares no LPIPS 360 (0.0 a 1.0)')
    
    args = parser.parse_args()
    
    # Determina se deve usar GPU
    use_gpu = args.use_gpu and not args.force_cpu
    if args.force_cpu:
        print("Forçando uso de CPU (GPU desabilitada)")
        # Desabilita GPU globalmente
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif use_gpu:
        print("Tentando usar GPU se disponível")
    else:
        print("Usando CPU (padrão)")
    
    all_results = []
    
    # Analisa cada método especificado
    for method in args.methods:
        method_dir = os.path.join(args.base_dir, method)
        
        if not os.path.exists(method_dir):
            print(f"Aviso: Diretório não encontrado: {method_dir}")
            continue
        
        # Busca subpastas (ex: hific-hi, hific-lo, hific-mi)
        subfolders = [d for d in os.listdir(method_dir) 
                     if os.path.isdir(os.path.join(method_dir, d))]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(method_dir, subfolder)
            compression_method = f"{method}-{subfolder}"
            
            print(f"\nAnalisando: {compression_method}")
            result = analyze_compression_folder(
                subfolder_path, compression_method, args.metrics, use_gpu,
                args.lpips360_weight_type, args.lpips360_pole_weight
            )
            
            if result:
                all_results.append(result)
    
    if all_results:
        print(f"\nResumo da análise:")
        for result in all_results:
            bpp_mean = np.mean(result['bpp_values']) if result['bpp_values'] else 0
            
            summary_parts = [f"BPP médio = {bpp_mean:.4f}"]
            
            if 'psnr_values' in result and result['psnr_values']:
                psnr_mean = np.mean(result['psnr_values'])
                summary_parts.append(f"PSNR médio = {psnr_mean:.2f} dB")
            
            if 'lpips_values' in result and result['lpips_values']:
                lpips_mean = np.mean(result['lpips_values'])
                summary_parts.append(f"LPIPS médio = {lpips_mean:.4f}")
            
            if 'lpips360_values' in result and result['lpips360_values']:
                lpips360_mean = np.mean(result['lpips360_values'])
                summary_parts.append(f"LPIPS360 médio = {lpips360_mean:.4f}")
            
            print(f"{result['method']}: {', '.join(summary_parts)}")
        
        plot_compression_analysis(all_results, args.metrics, args.output_dir)
    else:
        print("Nenhum resultado válido encontrado.")


if __name__ == '__main__':
    main()