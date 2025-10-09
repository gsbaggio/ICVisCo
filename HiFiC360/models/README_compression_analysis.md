# Análise de Compressão de Imagens

Este conjunto de scripts permite analisar e comparar diferentes métodos de compressão de imagens, plotando gráficos de rate-distortion (BPP vs métricas de qualidade).

## Arquivos

- `compression_analysis_standalone.py`: Script principal para análise
- `usage_examples.py`: Exemplos de uso e documentação
- `compression_analysis.py`: Versão original (requer dependências do HiFiC)

## Instalação de Dependências

```bash
pip install numpy pillow matplotlib
# Opcional para SSIM:
pip install scikit-image
```

## Uso Básico

```bash
# Análise básica com PSNR
python compression_analysis_standalone.py --base_dir files --methods hific --metrics psnr

# Múltiplas métricas
python compression_analysis_standalone.py --methods hific --metrics psnr mse --output_dir results

# Ver exemplos de uso
python usage_examples.py
```

## Estrutura de Diretórios Esperada

```
files/
├── hific/
│   ├── hific-hi/
│   │   ├── original/     # Imagens originais (.png, .jpg)
│   │   ├── compressed/   # Arquivos comprimidos (.tfci)
│   │   └── decompressed/ # Imagens decomprimidas (.png)
│   ├── hific-lo/
│   │   └── ... (mesma estrutura)
│   └── hific-mi/
│       └── ... (mesma estrutura)
└── outro_metodo/  # Para adicionar novos métodos
    ├── config1/
    │   └── ... (mesma estrutura)
    └── config2/
        └── ... (mesma estrutura)
```

## Métricas Disponíveis

- **PSNR** (Peak Signal-to-Noise Ratio): Qualidade da imagem em dB
- **MSE** (Mean Squared Error): Erro médio quadrático
- **SSIM** (Structural Similarity Index): Similaridade estrutural (requer scikit-image)

## Como Funciona

1. **Cálculo de BPP**: Compara tamanho do arquivo comprimido (.tfci) com dimensões da imagem original
2. **Métricas de Qualidade**: Compara imagem original com decomprimida
3. **Visualização**: Gera gráficos scatter conectados mostrando trade-off compressão vs qualidade

## Exemplo de Saída

```
================================================================================
RESUMO DA ANÁLISE DE COMPRESSÃO
================================================================================
Método          BPP Médio    BPP Std      PSNR Médio   PSNR Std    
-------------------------------------------------------------------
hific-hific-hi  0.2941       0.0000       36.407       0.000       
hific-hific-lo  0.1315       0.0000       32.151       0.000       
hific-hific-mi  0.2305       0.0000       34.433       0.000       
================================================================================
```

## Adicionando Novos Métodos

1. Crie a estrutura de diretórios para o novo método
2. Garante que os arquivos sigam a convenção de nomenclatura
3. Execute o script incluindo o novo método: `--methods hific novo_metodo`

## Limitações Atuais

- LPIPS requer configuração adicional do TensorFlow
- SSIM requer scikit-image
- Assume formato .tfci para arquivos comprimidos
- Requer correspondência exata de nomes entre original/comprimido/decomprimido

## Extensões Futuras

- Suporte para outros formatos de compressão
- Implementação completa do LPIPS
- Métricas perceptuais adicionais
- Análise estatística mais avançada