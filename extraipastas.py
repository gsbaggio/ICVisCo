#!/usr/bin/env python3
"""
Script para extrair imagens JPG de diretórios aninhados e movê-las para o diretório raiz.

Estrutura esperada:
    base_path/
        pano1024x512/
            indoor/
                subdir1/
                    image1.jpg
                    image2.jpg
                subdir2/
                    image3.jpg
            outdoor/
                subdir3/
                    image4.jpg

Resultado:
    base_path/
        image1.jpg
        image2.jpg
        image3.jpg
        image4.jpg
        pano1024x512/ (permanece vazia)
"""

import os
import shutil
import argparse
from pathlib import Path


def extrair_imagens(base_path):
    """
    Extrai todas as imagens .jpg dos subdiretórios de pano1024x512/indoor e pano1024x512/outdoor
    e move para o diretório base.
    
    Args:
        base_path (str): Caminho para o diretório base (ex: /home/tensorflow_datasets/SUN360)
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"❌ Erro: O diretório '{base_path}' não existe.")
        return
    
    pano_dir = base_path / "pano1024x512"
    
    if not pano_dir.exists():
        print(f"❌ Erro: O diretório '{pano_dir}' não existe.")
        return
    
    # Contadores
    total_movidos = 0
    total_erros = 0
    
    # Processar indoor e outdoor
    for subdir_name in ["indoor", "outdoor"]:
        subdir_path = pano_dir / subdir_name
        
        if not subdir_path.exists():
            print(f"⚠️  Aviso: Diretório '{subdir_path}' não encontrado. Pulando...")
            continue
        
        print(f"\n📂 Processando: {subdir_name}/")
        
        # Buscar todas as imagens .jpg recursivamente
        jpg_files = list(subdir_path.rglob("*.jpg")) + list(subdir_path.rglob("*.JPG"))
        
        if not jpg_files:
            print(f"   ℹ️  Nenhuma imagem .jpg encontrada em {subdir_name}/")
            continue
        
        print(f"   Encontradas {len(jpg_files)} imagens")
        
        # Mover cada imagem para o diretório base
        for img_path in jpg_files:
            try:
                # Nome do arquivo de destino
                destino = base_path / img_path.name
                
                # Se já existe um arquivo com o mesmo nome, adicionar sufixo
                if destino.exists():
                    base_name = img_path.stem
                    extension = img_path.suffix
                    contador = 1
                    while destino.exists():
                        destino = base_path / f"{base_name}_{contador}{extension}"
                        contador += 1
                    print(f"   ⚠️  Arquivo '{img_path.name}' já existe. Renomeando para '{destino.name}'")
                
                # Mover arquivo
                shutil.move(str(img_path), str(destino))
                total_movidos += 1
                
            except Exception as e:
                print(f"   ❌ Erro ao mover '{img_path.name}': {e}")
                total_erros += 1
    
    # Resumo
    print(f"\n{'='*60}")
    print(f"✅ Total de imagens movidas: {total_movidos}")
    if total_erros > 0:
        print(f"❌ Total de erros: {total_erros}")
    print(f"{'='*60}")
    
    # Verificar se os diretórios ficaram vazios
    print(f"\n📊 Verificando diretórios vazios...")
    for subdir_name in ["indoor", "outdoor"]:
        subdir_path = pano_dir / subdir_name
        if subdir_path.exists():
            remaining_files = list(subdir_path.rglob("*.*"))
            if not remaining_files:
                print(f"   ✓ {subdir_name}/ está vazio")
            else:
                print(f"   ℹ️  {subdir_name}/ ainda contém {len(remaining_files)} arquivo(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Extrai imagens JPG de diretórios aninhados e move para o diretório raiz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo de uso:
    python extraipastas.py --path /home/tensorflow_datasets/SUN360
    python extraipastas.py -p /home/tensorflow_datasets/SUN360
        """
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Caminho para o diretório base (ex: /home/tensorflow_datasets/SUN360)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula a operação sem mover os arquivos"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 Modo DRY-RUN ativado - Nenhum arquivo será movido\n")
        # Implementar dry-run se necessário
        print("⚠️  Funcionalidade dry-run não implementada ainda.")
        return
    
    print(f"🚀 Iniciando extração de imagens de: {args.path}\n")
    extrair_imagens(args.path)
    print(f"\n✨ Processo concluído!")


if __name__ == "__main__":
    main()
