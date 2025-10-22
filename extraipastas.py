import os
import shutil
import argparse
from pathlib import Path


def extrair_imagens(base_path):
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"O diretório '{base_path}' não existe.")
        return
    
    pano_dir = base_path / "pano1024x512"
    
    if not pano_dir.exists():
        print(f"O diretório '{pano_dir}' não existe.")
        return
    
    total_movidos = 0
    total_erros = 0
    
    for subdir_name in ["indoor", "outdoor"]:
        subdir_path = pano_dir / subdir_name
        
        if not subdir_path.exists():
            print(f"Diretório '{subdir_path}' não encontrado. Pulando...")
            continue
        
        print(f"\nProcessando: {subdir_name}/")
        
        jpg_files = list(subdir_path.rglob("*.jpg")) + list(subdir_path.rglob("*.JPG"))
        
        if not jpg_files:
            print(f"Nenhuma imagem .jpg encontrada em {subdir_name}/")
            continue
        
        print(f"Encontradas {len(jpg_files)} imagens")
        
        for img_path in jpg_files:
            try:
                destino = base_path / img_path.name
                
                if destino.exists():
                    base_name = img_path.stem
                    extension = img_path.suffix
                    contador = 1
                    while destino.exists():
                        destino = base_path / f"{base_name}_{contador}{extension}"
                        contador += 1
                    print(f"Arquivo '{img_path.name}' já existe. Renomeando para '{destino.name}'")
                
                shutil.move(str(img_path), str(destino))
                total_movidos += 1
                
            except Exception as e:
                print(f"Erro ao mover '{img_path.name}': {e}")
                total_erros += 1
    
    print(f"\n{'='*60}")
    print(f"Total de imagens movidas: {total_movidos}")
    if total_erros > 0:
        print(f"Total de erros: {total_erros}")
    print(f"{'='*60}")
    
    print(f"\nVerificando diretórios vazios...")
    for subdir_name in ["indoor", "outdoor"]:
        subdir_path = pano_dir / subdir_name
        if subdir_path.exists():
            remaining_files = list(subdir_path.rglob("*.*"))
            if not remaining_files:
                print(f"{subdir_name}/ está vazio")
            else:
                print(f"{subdir_name}/ ainda contém {len(remaining_files)} arquivo(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Extrai imagens JPG de diretórios aninhados e move para o diretório raiz.",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        print("Modo DRY-RUN ativado - Nenhum arquivo será movido\n")
        print("Funcionalidade dry-run não implementada ainda.")
        return

    print(f"Iniciando extração de imagens de: {args.path}\n")
    extrair_imagens(args.path)
    print(f"\nProcesso concluído!")


if __name__ == "__main__":
    main()
