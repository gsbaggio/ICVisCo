# HiFiC 360-Degree Image Compression

Este diretório contém uma versão adaptada do HiFiC para compressão de imagens 360 graus (equiretangulares) com uma loss function LPIPS adaptada que considera pesos baseados na latitude.

## Novas Funcionalidades

### 1. LPIPS Loss Adaptada para Imagens 360 (lpips_360.py)

A nova implementação inclui:

- **Pesos baseados na latitude**: Reduz a importância das regiões polares onde há maior distorção na projeção equiretangular
- **Tipos de peso configuráveis**: 
  - `cosine`: Peso baseado em cos(latitude) - mais natural para imagens 360
  - `linear`: Peso linear das regiões polares para o equador
  - `quadratic`: Peso quadrático - transição mais suave
- **Fator de peso polar**: Controla o peso mínimo das regiões polares (0.0 a 1.0)

### 2. Configurações Específicas para Imagens 360

Novas configurações otimizadas:
- `hific-360`: HiFiC com GAN e LPIPS 360
- `mselpips-360`: MSE + LPIPS 360 sem GAN

## Como Usar

### Treinamento Básico com LPIPS 360

```bash
python train.py \
  --config hific-360 \
  --ckpt_dir ./checkpoints/hific360_experiment \
  --num_steps 100k \
  --batch_size 4 \
  --crop_size 512 \
  --use_lpips_360 \
  --latitude_weight_type cosine \
  --pole_weight 0.3 \
  --tfds_data_dir /path/to/360/images
```

### Parâmetros da LPIPS 360

- `--use_lpips_360`: Habilita a loss LPIPS adaptada para 360°
- `--latitude_weight_type`: Tipo de peso por latitude
  - `cosine` (recomendado): Peso baseado em cos(latitude)
  - `linear`: Peso linear
  - `quadratic`: Peso quadrático
- `--pole_weight`: Peso mínimo para regiões polares (0.0-1.0)
  - `0.0`: Ignora completamente os polos
  - `0.5`: Peso médio nos polos
  - `1.0`: Peso uniforme (equivale ao LPIPS padrão)

### Script de Exemplo

Use o script `train_360_example.py` para facilitar o treinamento:

```bash
python train_360_example.py \
  --config hific-360 \
  --ckpt_dir ./checkpoints/my_360_model \
  --data_dir /path/to/360/dataset \
  --latitude_weight_type cosine \
  --pole_weight 0.3
```

## Teoria por Trás da Implementação

### Problema com Imagens 360

Imagens 360° em projeção equiretangular sofrem de distorção significativa nas regiões polares:
- **Stretching polar**: Pixels próximos aos polos representam áreas menores no mundo real
- **Importância perceptual**: Humanos focam mais nas regiões centrais (equatoriais)
- **Artifacts de compressão**: Distorções nos polos são menos perceptíveis

### Solução: Pesos por Latitude

A nova loss function aplica pesos w(φ) baseados na latitude φ:

```
LPIPS_360 = Σ w(φ) * LPIPS_local(φ)
```

Onde:
- **Cosine weighting**: w(φ) = cos(φ) * (1 - p) + p
- **Linear weighting**: w(φ) = 1 - |φ|/(π/2) * (1 - p)
- **Quadratic weighting**: w(φ) = 1 - (φ/(π/2))² * (1 - p)

O parâmetro `p` (pole_weight) controla o peso mínimo dos polos.

### Vantagens

1. **Melhor qualidade perceptual**: Foca nas regiões mais importantes
2. **Redução de artifacts**: Menos artifacts visíveis nas regiões polares
3. **Eficiência**: Melhor uso da taxa de bits disponível
4. **Flexibilidade**: Configurável para diferentes tipos de conteúdo

## Configurações Recomendadas

### Para Conteúdo Geral 360°
```bash
--latitude_weight_type cosine --pole_weight 0.3
```

### Para Conteúdo com Foco Equatorial
```bash
--latitude_weight_type cosine --pole_weight 0.1
```

### Para Validação/Comparação com LPIPS Padrão
```bash
--latitude_weight_type linear --pole_weight 1.0
```

## Estrutura de Arquivos Modificados

```
HiFiC360/models/hific/
├── lpips_360.py              # Nova implementação LPIPS 360
├── model.py                  # Modificado para suportar LPIPS 360
├── train.py                  # Argumentos adicionais para LPIPS 360
├── configs.py                # Novas configurações para 360°
├── train_360_example.py      # Script de exemplo
└── README_360.md             # Esta documentação
```

## Exemplo de Dataset

Para treinar com imagens 360°, organize seus dados em formato equiretangular:

```
dataset/
├── train/
│   ├── image_001.jpg  # 2:1 aspect ratio (e.g., 2048x1024)
│   ├── image_002.jpg
│   └── ...
└── validation/
    ├── val_001.jpg
    └── ...
```

## Monitoramento do Treinamento

Durante o treinamento, monitore:
- `weighted_lpips`: Loss LPIPS com peso por latitude
- `components/weighted_D`: Distorção ponderada
- `components/weighted_R`: Taxa ponderada

## Troubleshooting

### Problema: "Could not compute spatial LPIPS map"
**Solução**: O modelo volta automaticamente para LPIPS padrão com pesos aproximados.

### Problema: Qualidade ruim nos polos
**Solução**: Aumente o `pole_weight` para dar mais importância às regiões polares.

### Problema: Taxa de bits muito alta
**Solução**: Ajuste o `target` na configuração ou reduza `lpips_weight`.

## Citação

Se usar esta implementação, por favor cite:

```bibtex
@misc{hific360,
  title={HiFiC 360: High-Fidelity Compression for 360-degree Images},
  author={Gabriel Baggio},
  year={2024},
  note={Adaptação do HiFiC com LPIPS baseada em latitude para imagens 360°}
}
```