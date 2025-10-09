# 🎛️ Configurações HiFiC 360 - Guia de Compressão

## 📊 Configurações Disponíveis

### 🔥 **Modelos HiFiC com GAN (Melhor Qualidade Perceptual)**

| Configuração | Taxa de Bits | Qualidade | Uso Recomendado |
|-------------|--------------|-----------|-----------------|
| `hific-360-lo` | ~0.10 bpp | Baixa | Streaming, armazenamento |
| `hific-360-mi` | ~0.20 bpp | Média | **Uso geral** (recomendado) |
| `hific-360-hi` | ~0.35 bpp | Alta | Arquivo, produção |
| `hific-360` | ~0.16 bpp | Balanceada | Teste/desenvolvimento |

### 🧮 **Modelos MSE+LPIPS (Sem GAN, Mais Rápido)**

| Configuração | Taxa de Bits | Qualidade | Uso Recomendado |
|-------------|--------------|-----------|-----------------|
| `mselpips-360-lo` | ~0.12 bpp | Baixa | Prototipagem rápida |
| `mselpips-360` | ~0.16 bpp | Média | Teste sem GAN |
| `mselpips-360-hi` | ~0.30 bpp | Alta | Qualidade sem GAN |

## 🎯 **Parâmetros-Chave por Configuração**

### **Low (LO) - Máxima Compressão**
```python
target=0.10-0.12 bpp        # Taxa de bits muito baixa
CD=0.6                      # Menor peso na distorção
lpips_weight=1.0            # LPIPS padrão
lmbda_a=2^-7               # Penalidade forte na taxa
```

### **Medium (MI) - Balanceado** ⭐ *Recomendado*
```python
target=0.20 bpp            # Taxa balanceada
CD=0.65                    # Peso balanceado na distorção  
lpips_weight=1.2           # LPIPS melhorado
lmbda_a=2^-6              # Penalidade média na taxa
```

### **High (HI) - Máxima Qualidade**
```python
target=0.30-0.35 bpp       # Taxa de bits alta
CD=0.75                    # Máximo peso na distorção
lpips_weight=1.4-1.5       # LPIPS maximizado
lmbda_a=2^-5              # Penalidade relaxada na taxa
```

## 🚀 **Como Usar**

### Comando Direto:
```bash
# Baixa compressão (máxima qualidade)
python train.py --config hific-360-hi --use_lpips_360 --ckpt_dir ./checkpoints/hi_quality

# Compressão média (recomendado)
python train.py --config hific-360-mi --use_lpips_360 --ckpt_dir ./checkpoints/medium

# Alta compressão (menor tamanho)
python train.py --config hific-360-lo --use_lpips_360 --ckpt_dir ./checkpoints/compressed
```

### Script Rápido:
```bash
# Máxima qualidade
./quick_train_360.sh /path/to/images hific-360-hi

# Balanceado (padrão)
./quick_train_360.sh /path/to/images hific-360-mi

# Máxima compressão
./quick_train_360.sh /path/to/images hific-360-lo
```

## 📈 **Personalizando Configurações**

### Para criar sua própria configuração:

1. **Edite `configs.py`** e adicione:
```python
'minha-config-360': helpers.Config(
    model_type=helpers.ModelType.COMPRESSION_GAN,
    lambda_schedule=helpers.Config(vals=[2., 1.], steps=[50000]),
    lr=8e-5,
    lr_schedule=helpers.Config(vals=[1., 0.1], steps=[500000]),
    num_steps_disc=1,
    loss_config=helpers.Config(
        CP=0.1 * 1.5 ** 1,
        C=0.1 * 2. ** -5,
        CD=0.70,              # Ajuste aqui: 0.6-0.8
        target=0.25,          # Ajuste aqui: 0.08-0.50
        lpips_weight=1.3,     # Ajuste aqui: 1.0-1.5
        target_schedule=helpers.Config(vals=[0.35/0.25, 1.], steps=[50000]),
        lmbda_a=0.1 * 2. ** -6,  # Ajuste aqui: 2^-8 a 2^-4
        lmbda_b=0.1 * 2. ** 1,
    )
)
```

2. **Use a nova configuração**:
```bash
python train.py --config minha-config-360 --use_lpips_360
```

## 🎛️ **Ajuste Fino de Parâmetros**

### **Para MAIS compressão (arquivo menor):**
- ⬇️ Diminua `target` (ex: 0.08, 0.10)
- ⬇️ Diminua `CD` (ex: 0.55, 0.60)
- ⬇️ Diminua `lmbda_a` (ex: 2^-8, 2^-7)

### **Para MELHOR qualidade (arquivo maior):**
- ⬆️ Aumente `target` (ex: 0.40, 0.50)
- ⬆️ Aumente `CD` (ex: 0.80, 0.85)
- ⬆️ Aumente `lpips_weight` (ex: 1.6, 1.8)
- ⬆️ Aumente `lmbda_a` (ex: 2^-4, 2^-3)

### **Para imagens 360° específicas:**
- 🌍 **Paisagens**: Use `hific-360-mi` com `--pole_weight 0.3`
- 🏃 **Esportes/Ação**: Use `hific-360-hi` com `--pole_weight 0.1`
- 🎨 **Arte/Cinema**: Use `hific-360-hi` com `--pole_weight 0.4`

## 📊 **Comparação de Resultados Esperados**

| Config | Tamanho do Arquivo | Qualidade Visual | Tempo de Treinamento |
|--------|-------------------|------------------|---------------------|
| `hific-360-lo` | **Menor** | Boa | Rápido |
| `hific-360-mi` | Médio | **Muito Boa** | Médio |
| `hific-360-hi` | Maior | **Excelente** | Mais lento |

## 🔧 **Dicas de Uso**

1. **Para desenvolvimento**: Comece com `hific-360-mi`
2. **Para produção**: Use `hific-360-hi` se o tamanho não for crítico
3. **Para streaming**: Use `hific-360-lo` 
4. **Para arquivos**: Use `hific-360-hi`
5. **Sem GPU potente**: Use configurações `mselpips-360-*`

## 🎯 **Comandos Prontos**

```bash
# Desenvolvimento/Teste (meio termo)
./quick_train_360.sh /path/to/images hific-360-mi

# Produção de Alta Qualidade
./quick_train_360.sh /path/to/images hific-360-hi

# Streaming/Compressão Máxima
./quick_train_360.sh /path/to/images hific-360-lo

# Teste Rápido (sem GAN)
./quick_train_360.sh /path/to/images mselpips-360
```