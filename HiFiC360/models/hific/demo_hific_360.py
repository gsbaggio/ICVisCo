#!/usr/bin/env python3
"""
Demonstração da funcionalidade LPIPS 360 - HiFiC para imagens 360°.

Este script demonstra como usar a nova loss function LPIPS adaptada
para imagens 360° com pesos baseados na latitude.
"""

import os
import sys
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
compression_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
    'compression', 'models'
)
sys.path.insert(0, current_dir)
sys.path.insert(0, compression_path)

def demonstrate_latitude_weights():
    """Demonstrar como os pesos por latitude funcionam."""
    print("🌍 DEMONSTRAÇÃO: Pesos por Latitude para Imagens 360°")
    print("=" * 60)
    
    # Simular coordenadas de latitude para uma imagem 360°
    height = 180  # pixels
    lat_degrees = np.linspace(-90, 90, height)  # -90° (polo sul) a +90° (polo norte)
    lat_radians = np.radians(lat_degrees)
    
    # Diferentes esquemas de peso
    pole_weight = 0.3
    
    # 1. Cosine weighting (recomendado para 360°)
    cosine_weights = np.cos(lat_radians) * (1.0 - pole_weight) + pole_weight
    
    # 2. Linear weighting
    abs_lat = np.abs(lat_radians)
    max_lat = np.pi / 2
    linear_weights = 1.0 - (abs_lat / max_lat) * (1.0 - pole_weight)
    
    # 3. Quadratic weighting
    normalized_lat = abs_lat / max_lat
    quadratic_weights = 1.0 - normalized_lat**2 * (1.0 - pole_weight)
    
    print(f"📊 Comparação de Pesos (pole_weight = {pole_weight}):")
    print(f"{'Latitude':<10} {'Cosine':<8} {'Linear':<8} {'Quadratic':<10}")
    print("-" * 40)
    
    for i in range(0, height, 30):  # Mostrar a cada 30°
        lat = lat_degrees[i]
        cos_w = cosine_weights[i]
        lin_w = linear_weights[i]
        quad_w = quadratic_weights[i]
        print(f"{lat:8.0f}°   {cos_w:.3f}    {lin_w:.3f}    {quad_w:.3f}")
    
    print("\n🎯 Interpretação:")
    print("- Valores maiores = mais importância na loss function")
    print("- Equador (0°) tem maior peso = melhor qualidade")
    print("- Polos (±90°) têm menor peso = menos artifacts visíveis")
    
    return cosine_weights, linear_weights, quadratic_weights


def demonstrate_360_loss_concept():
    """Demonstrar o conceito da loss LPIPS 360."""
    print("\n🧠 CONCEITO: Como a LPIPS 360 Funciona")
    print("=" * 60)
    
    print("1. 📐 PROBLEMA com imagens 360°:")
    print("   - Projeção equiretangular distorce regiões polares")
    print("   - Pixels nos polos representam menos área real")
    print("   - Humanos focam mais nas regiões centrais")
    
    print("\n2. 💡 SOLUÇÃO LPIPS 360:")
    print("   - Aplica pesos W(φ) baseados na latitude φ")
    print("   - Loss = Σ W(φ) × LPIPS_local(φ)")
    print("   - Reduz importância dos polos automaticamente")
    
    print("\n3. 🎛️ CONFIGURAÇÕES:")
    print("   - cosine: W(φ) = cos(φ) × (1-p) + p")
    print("   - linear: W(φ) = 1 - |φ|/(π/2) × (1-p)")
    print("   - quadratic: W(φ) = 1 - (φ/(π/2))² × (1-p)")
    print("   onde p = pole_weight")
    
    print("\n4. 🎯 BENEFÍCIOS:")
    print("   ✅ Melhor qualidade perceptual")
    print("   ✅ Menos artifacts nos polos")
    print("   ✅ Uso eficiente de bits")
    print("   ✅ Configurável por tipo de conteúdo")


def demonstrate_usage_examples():
    """Demonstrar exemplos de uso."""
    print("\n🚀 EXEMPLOS DE USO")
    print("=" * 60)
    
    print("📺 Para conteúdo 360° geral (paisagens, ambientes):")
    print("   --latitude_weight_type cosine --pole_weight 0.3")
    print("   → Balanceado, foco no equador com polos suavizados")
    
    print("\n🎬 Para conteúdo com ação no equador (esportes, eventos):")
    print("   --latitude_weight_type cosine --pole_weight 0.1")
    print("   → Máximo foco no equador, polos minimizados")
    
    print("\n🔬 Para análise/comparação científica:")
    print("   --latitude_weight_type linear --pole_weight 1.0")
    print("   → Peso uniforme (equivale ao LPIPS padrão)")
    
    print("\n🎨 Para conteúdo artístico com gradações suaves:")
    print("   --latitude_weight_type quadratic --pole_weight 0.4")
    print("   → Transição gradual, preserva detalhes médios")


def test_import_functionality():
    """Testar se a funcionalidade pode ser importada."""
    print("\n🔧 TESTE DE FUNCIONALIDADE")
    print("=" * 60)
    
    try:
        # Testar import da LPIPS 360
        from lpips_360 import LPIPS360Loss, LPIPS360LossFactory
        print("✅ LPIPS360Loss importada com sucesso")
        
        # Testar configurações
        from hific import configs
        config_360 = configs.get_config('hific-360')
        print("✅ Configuração 'hific-360' carregada")
        print(f"   - LPIPS weight: {config_360.loss_config.lpips_weight}")
        print(f"   - Learning rate: {config_360.lr}")
        
        # Testar factory
        loss_equator = LPIPS360LossFactory.create_equator_focused_loss("dummy_path", 0.2)
        print("✅ Factory methods funcionando")
        
        print("\n🎉 Todas as funcionalidades estão operacionais!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar funcionalidade: {e}")
        print("   Verifique se os paths estão configurados corretamente")
        return False


def show_next_steps():
    """Mostrar próximos passos."""
    print("\n📋 PRÓXIMOS PASSOS")
    print("=" * 60)
    
    print("1. 🗂️  PREPARAR DATASET:")
    print("   - Organize imagens 360° em formato equiretangular")
    print("   - Proporção 2:1 (ex: 2048x1024, 4096x2048)")
    print("   - Formato: dataset/train/, dataset/validation/")
    
    print("\n2. ⚙️  CONFIGURAR AMBIENTE:")
    print("   conda activate hific")
    print("   export PYTHONPATH=\"$(pwd):/path/to/compression/models:$PYTHONPATH\"")
    
    print("\n3. 🚀 EXECUTAR TREINAMENTO:")
    print("   python train.py \\")
    print("     --config hific-360 \\")
    print("     --ckpt_dir ./checkpoints \\")
    print("     --use_lpips_360 \\")
    print("     --latitude_weight_type cosine \\")
    print("     --pole_weight 0.3")
    
    print("\n4. 📊 MONITORAR MÉTRICAS:")
    print("   - weighted_lpips: Loss com pesos por latitude")
    print("   - components/weighted_D: Distorção ponderada")
    print("   - components/weighted_R: Taxa ponderada")
    
    print("\n5. 🔬 EXPERIMENTAR:")
    print("   - Teste diferentes pole_weight (0.1, 0.3, 0.5)")
    print("   - Compare latitude_weight_type (cosine, linear, quadratic)")
    print("   - Analise qualidade vs taxa de bits")


def main():
    """Função principal da demonstração."""
    print("🌟 HiFiC 360 - Compressão de Imagens 360° com LPIPS Adaptada")
    print("=" * 70)
    print("Implementação de Gabriel Baggio - Outubro 2024")
    print("Baseado no HiFiC original (Google, 2020)")
    
    # Demonstrações
    demonstrate_latitude_weights()
    demonstrate_360_loss_concept()
    demonstrate_usage_examples()
    
    # Teste de funcionalidade
    if test_import_functionality():
        show_next_steps()
        
        print("\n" + "=" * 70)
        print("🎊 IMPLEMENTAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 70)
        print("✨ Sua versão do HiFiC agora suporta imagens 360° com LPIPS adaptada!")
        print("✨ Use os exemplos acima para começar a treinar seus modelos.")
        print("✨ Para mais detalhes, consulte README_360.md e USAGE_GUIDE.md")
    else:
        print("\n" + "=" * 70)
        print("⚠️  CONFIGURAÇÃO NECESSÁRIA")
        print("=" * 70)
        print("Configure os paths do Python conforme mostrado no USAGE_GUIDE.md")


if __name__ == '__main__':
    main()