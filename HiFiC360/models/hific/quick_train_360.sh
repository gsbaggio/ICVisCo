#!/bin/bash
# Script rápido para treinar HiFiC 360
# Usage: ./quick_train_360.sh /path/to/images [config_name]

if [ $# -eq 0 ]; then
    echo "Uso: $0 /caminho/para/imagens [configuracao]"
    echo ""
    echo "Configurações disponíveis:"
    echo "  hific-360-lo    - Baixa taxa de bits (~0.10 bpp)"
    echo "  hific-360-mi    - Taxa média (~0.20 bpp) [PADRÃO]"
    echo "  hific-360-hi    - Alta taxa de bits (~0.35 bpp)"
    echo "  hific-360       - Configuração balanceada (~0.16 bpp)"
    echo "  mselpips-360-lo - MSE+LPIPS baixa taxa (~0.12 bpp)"
    echo "  mselpips-360    - MSE+LPIPS balanceada (~0.16 bpp)"
    echo "  mselpips-360-hi - MSE+LPIPS alta taxa (~0.30 bpp)"
    echo ""
    echo "Exemplo: $0 /home/usuario/dataset_360 hific-360-mi"
    echo ""
    echo "IMPORTANTE: Execute com 'source' para ativar conda:"
    echo "  source ./quick_train_360.sh /path/to/images hific-360-mi"
    exit 1
fi

DATASET_DIR="$1"
CONFIG_NAME="${2:-hific-360-mi}"  # Default to medium quality
CHECKPOINT_DIR="./checkpoints/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Iniciando treinamento HiFiC 360..."
echo "📁 Dataset: $DATASET_DIR"
echo "⚙️  Configuração: $CONFIG_NAME"
echo "💾 Checkpoints: $CHECKPOINT_DIR"

# Verificar se a configuração existe
case "$CONFIG_NAME" in
    hific-360-lo|hific-360-mi|hific-360-hi|hific-360|mselpips-360-lo|mselpips-360|mselpips-360-hi)
        echo "✅ Configuração válida: $CONFIG_NAME"
        ;;
    *)
        echo "❌ Configuração inválida: $CONFIG_NAME"
        echo "Use uma das configurações listadas no help acima."
        exit 1
        ;;
esac

# Inicializar conda se necessário
if ! command -v conda &> /dev/null; then
    echo "⚙️  Inicializando conda..."
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Ativar ambiente hific
echo "🔄 Ativando ambiente conda 'hific'..."
conda activate hific

# Verificar se o ambiente foi ativado
if [[ "$CONDA_DEFAULT_ENV" != "hific" ]]; then
    echo "❌ Falha ao ativar ambiente 'hific'"
    echo "Execute manualmente: conda activate hific"
    exit 1
fi

echo "✅ Ambiente conda 'hific' ativado"

# Configurar paths
export PYTHONPATH="$(pwd):/home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/compression/models:$PYTHONPATH"
echo "🔧 PYTHONPATH configurado"

# Executar treinamento
echo "🎯 Iniciando treinamento..."
python train.py \
  --config "$CONFIG_NAME" \
  --ckpt_dir "$CHECKPOINT_DIR" \
  --num_steps 5k \
  --batch_size 4 \
  --crop_size 512 \
  --use_lpips_360 \
  --latitude_weight_type cosine \
  --pole_weight 0.3 \
  --tfds_dataset_name image_folder \
  --tfds_downloads_dir "$DATASET_DIR" \
  --tfds_features_key image

echo "✅ Treinamento concluído!"
echo "📊 Checkpoints salvos em: $CHECKPOINT_DIR"