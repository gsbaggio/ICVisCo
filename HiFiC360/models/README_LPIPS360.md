# LPIPS 360 - Perceptual Loss com Pesos por Latitude

Esta implementação estende o LPIPS tradicional com um sistema de pesos por latitude especificamente projetado para análise de imagens 360 graus em projeção equiretangular.

## Características Principais

### Ponderação por Latitude
- **Objetivo**: Compensar distorções naturais da projeção equiretangular
- **Conceito**: Pólos são naturalmente distorcidos, equador preserva geometria
- **Implementação**: Aplica pesos diferentes baseados na posição de latitude

### Tipos de Peso Disponíveis

#### 1. Coseno (`cosine`) - **Recomendado**
```bash
--lpips360_weight_type cosine --lpips360_pole_weight 0.3
```
- **Comportamento**: Peso máximo no equador, redução natural nos pólos
- **Fórmula**: `cos(latitude) * (1 - pole_weight) + pole_weight`
- **Uso recomendado**: Análise geral de imagens 360, compensação natural da distorção

#### 2. Linear (`linear`)
```bash
--lpips360_weight_type linear --lpips360_pole_weight 0.5
```
- **Comportamento**: Transição linear do equador aos pólos
- **Fórmula**: `1 - |latitude|/(π/2) * (1 - pole_weight)`
- **Uso recomendado**: Análise onde distorção polar deve ser moderadamente reduzida

#### 3. Quadrático (`quadratic`)
```bash
--lpips360_weight_type quadratic --lpips360_pole_weight 0.7
```
- **Comportamento**: Transição suave com concentração no equador
- **Fórmula**: `1 - (latitude/(π/2))² * (1 - pole_weight)`
- **Uso recomendado**: Análise com foco extremo no equador

### Parâmetro Pole Weight

O `pole_weight` controla o peso mínimo nas regiões polares:

- **0.0**: Pólos completamente ignorados (não recomendado)
- **0.3**: Pólos têm 30% do peso do equador (**recomendado para coseno**)
- **0.5**: Pólos têm 50% do peso do equador (balanceado)
- **0.8**: Pólos têm 80% do peso do equador (quase uniforme)
- **1.0**: Peso uniforme (equivale ao LPIPS tradicional)

## Exemplos de Uso

### Análise Básica com LPIPS 360
```bash
python compression_analysis.py \
    --base_dir files \
    --methods hific \
    --metrics psnr lpips360 \
    --force_cpu \
    --lpips360_weight_type cosine \
    --lpips360_pole_weight 0.3
```

### Comparação LPIPS Tradicional vs 360
```bash
python compression_analysis.py \
    --base_dir files \
    --methods hific \
    --metrics psnr lpips lpips360 \
    --force_cpu \
    --lpips360_weight_type cosine \
    --lpips360_pole_weight 0.3
```

### Análise com Diferentes Tipos de Peso
```bash
# Peso coseno (natural)
python compression_analysis.py --metrics lpips360 --lpips360_weight_type cosine --lpips360_pole_weight 0.3

# Peso linear (moderado)
python compression_analysis.py --metrics lpips360 --lpips360_weight_type linear --lpips360_pole_weight 0.5

# Peso quadrático (concentrado)
python compression_analysis.py --metrics lpips360 --lpips360_weight_type quadratic --lpips360_pole_weight 0.7
```

## Resultados Típicos

### Diferenças entre LPIPS e LPIPS 360

Para imagens 360 de qualidade similar, observa-se:

| Método | LPIPS | LPIPS 360 (coseno) | Diferença |
|--------|-------|-------------------|-----------|
| Alta qualidade | 0.0068 | 0.0050 | -26% |
| Baixa qualidade | 0.0206 | 0.0154 | -25% |
| Qualidade média | 0.0100 | 0.0075 | -25% |

**Interpretação**: LPIPS 360 com peso coseno geralmente retorna valores menores que LPIPS tradicional, indicando que a ponderação por latitude reduz a contribuição das distorções polares naturais.

### Comparação entre Tipos de Peso

Para mesma imagem (baixa qualidade):

| Tipo | Pole Weight | LPIPS 360 | Característica |
|------|-------------|-----------|----------------|
| coseno | 0.3 | 0.0154 | Foco natural no equador |
| linear | 0.5 | 0.0155 | Transição equilibrada |
| quadrático | 0.8 | 0.0192 | Mais peso nos pólos |

## Implementação Técnica

### Arquitetura
- **Base**: LPIPS tradicional com AlexNet
- **Extensão**: Mapa de pesos por latitude aplicado às saídas
- **Compatibilidade**: TensorFlow 1.x, CPU/GPU
- **Integração**: Transparente com pipeline de análise existente

### Dependências
- TensorFlow 1.15.x
- NumPy
- Matplotlib (para visualização)
- Arquivo de pesos LPIPS: `lpips_weight__net-lin_alex_v0.1.pb`

### Arquivos Principais
- `simple_lpips360.py`: Implementação compatível com TF 1.x
- `compression_analysis.py`: Script integrado de análise
- `lpips_360.py`: Implementação original (com problemas de compatibilidade)

## Recomendações

### Para Análise Geral de Imagens 360
```bash
--lpips360_weight_type cosine --lpips360_pole_weight 0.3
```

### Para Estudos Comparativos
Inclua tanto LPIPS quanto LPIPS 360 para avaliar o impacto da ponderação:
```bash
--metrics psnr lpips lpips360
```

### Para Imagens com Conteúdo Polar Importante
Use pole_weight mais alto para preservar análise nas regiões polares:
```bash
--lpips360_weight_type cosine --lpips360_pole_weight 0.7
```

## Limitações

1. **Aproximação**: Aplica peso médio por latitude em vez de peso pixel-wise
2. **Projeção**: Projetado especificamente para equiretangular
3. **Compatibilidade**: Requer TensorFlow 1.x devido ao modelo base
4. **Performance**: Processamento adicional para cálculo de pesos

## Resultados Esperados

O LPIPS 360 deve fornecer uma medida perceptual mais precisa para imagens 360, reduzindo a influência de artefatos de distorção polar inerentes à projeção equiretangular e focando na qualidade percebida em regiões geometricamente importantes.