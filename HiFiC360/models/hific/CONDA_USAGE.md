# Como Usar o HiFiC 360 com Conda

## Método 1: Script Automático (Recomendado)

Execute o script com `source` para que o conda seja ativado corretamente:

```bash
cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models/hific
source ./quick_train_360.sh /caminho/para/suas/imagens hific-360-mi
```

O script irá:
1. ✅ Verificar e inicializar o conda automaticamente
2. ✅ Ativar o ambiente 'hific'
3. ✅ Configurar o PYTHONPATH
4. ✅ Executar o treinamento

## Método 2: Manual (Controle Total)

Se preferir fazer tudo manualmente:

```bash
# 1. Ativar o ambiente conda
conda activate hific

# 2. Ir para o diretório do HiFiC 360
cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models/hific

# 3. Configurar o Python path
export PYTHONPATH="$(pwd):/home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/compression/models:$PYTHONPATH"

# 4. Executar o treinamento
python train.py \
  --config hific-360-mi \
  --ckpt_dir ./checkpoints/meu_treinamento \
  --num_steps 5k \
  --batch_size 4 \
  --crop_size 512 \
  --use_lpips_360 \
  --latitude_weight_type cosine \
  --pole_weight 0.3 \
  --tfds_dataset_name image_folder \
  --tfds_downloads_dir /caminho/para/suas/imagens \
  --tfds_features_key image
```

## Método 3: Uma Linha Simples

Para um comando rápido:

```bash
conda activate hific && cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models/hific && source ./quick_train_360.sh /caminho/para/imagens hific-360-mi
```

## Verificar se o Ambiente Está Ativo

Para verificar se o conda está funcionando:

```bash
# Verificar ambiente ativo
echo $CONDA_DEFAULT_ENV
# Deve mostrar: hific

# Verificar pacotes instalados
conda list | grep tensorflow
# Deve mostrar tensorflow e outras dependências

# Testar importação Python
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

## Configurações Disponíveis

| Configuração     | Taxa (bpp) | Qualidade | Uso de RAM | Tempo  |
|------------------|------------|-----------|------------|--------|
| `hific-360-lo`   | ~0.10      | Básica    | Baixo      | Rápido |
| `hific-360-mi`   | ~0.20      | Média     | Médio      | Médio  |
| `hific-360-hi`   | ~0.35      | Alta      | Alto       | Lento  |
| `hific-360`      | ~0.16      | Balanceada| Médio      | Médio  |
| `mselpips-360-lo`| ~0.12      | MSE+LPIPS | Baixo      | Rápido |
| `mselpips-360`   | ~0.16      | MSE+LPIPS | Médio      | Médio  |
| `mselpips-360-hi`| ~0.30      | MSE+LPIPS | Alto       | Lento  |

## Estrutura do Dataset

Organize suas imagens 360° assim:

```
/caminho/para/dataset/
├── train/
│   ├── imagem1.jpg
│   ├── imagem2.jpg
│   └── ...
├── valid/
│   ├── val1.jpg
│   ├── val2.jpg
│   └── ...
└── test/
    ├── test1.jpg
    ├── test2.jpg
    └── ...
```

## Monitoramento

Durante o treinamento, você pode:

```bash
# Ver logs em tempo real
tail -f checkpoints/*/logs.txt

# Ver métricas
tensorboard --logdir checkpoints/

# Verificar uso de GPU
nvidia-smi -l 1
```