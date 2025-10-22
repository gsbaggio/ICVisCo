import os
import shutil
import random
import argparse
from pathlib import Path


def listar_imagens(diretorio):
    extensoes_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    imagens = []
    
    for arquivo in diretorio.iterdir():
        if arquivo.is_file() and arquivo.suffix in extensoes_validas:
            imagens.append(arquivo)
    
    return imagens


def copiar_imagens_aleatorias(source_dir, dest_dir, num_imagens, base_path=None):
    if base_path:
        base = Path(base_path)
    else:
        base = Path.home() / "tensorflow_datasets"
    
    source = base / source_dir
    dest = base / dest_dir
    
    if not source.exists():
        print(f"Diretório fonte '{source}' não existe.")
        return
    
    if not source.is_dir():
        print(f"{source}' não é um diretório.")
        return
    
    if not dest.exists():
        print(f"Criando diretório destino: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Diretório destino já existe: {dest}")
    
    imagens = listar_imagens(source)
    
    if not imagens:
        print(f"Nenhuma imagem encontrada em '{source}'")
        return
    
    print(f"Encontradas {len(imagens)} imagens no diretório fonte")
    
    if num_imagens > len(imagens):
        print(f"Solicitadas {num_imagens} imagens, mas só existem {len(imagens)}.")
        print(f"Copiando todas as {len(imagens)} imagens disponíveis.")
        num_imagens = len(imagens)
    
    print(f"Selecionando {num_imagens} imagens aleatórias...")
    imagens_selecionadas = random.sample(imagens, num_imagens)
    
    print(f"\nIniciando cópia de {num_imagens} imagens...")
    print(f"   Origem: {source}")
    print(f"   Destino: {dest}")
    print()
    
    copiadas = 0
    erros = 0
    
    for i, img_path in enumerate(imagens_selecionadas, 1):
        try:
            destino_arquivo = dest / img_path.name
            
            if destino_arquivo.exists():
                print(f"[{i}/{num_imagens}] '{img_path.name}' já existe no destino. Pulando...")
                continue
            
            shutil.copy2(img_path, destino_arquivo)
            copiadas += 1
            
            if i % 100 == 0 or i == num_imagens:
                print(f"Progresso: {i}/{num_imagens} ({(i/num_imagens)*100:.1f}%)")
            
        except Exception as e:
            print(f"Erro ao copiar '{img_path.name}': {e}")
            erros += 1
    
    # Resumo final
    print(f"\n{'='*60}")
    print(f"Cópia concluída!")
    print(f" Imagens copiadas: {copiadas}")
    if erros > 0:
        print(f"Erros: {erros}")
    print(f"Destino: {dest}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Copia um número específico de imagens aleatórias de um dataset para outro diretório.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Nome do diretório fonte (ex: SUN360)"
    )
    
    parser.add_argument(
        "--dest", "-d",
        type=str,
        required=True,
        help="Nome do diretório destino (ex: SUN360-2000)"
    )
    
    parser.add_argument(
        "--count", "-n",
        type=int,
        required=True,
        help="Número de imagens a copiar (ex: 2000)"
    )
    
    parser.add_argument(
        "--base", "-b",
        type=str,
        default=None,
        help="Caminho base onde estão os diretórios (padrão: ~/tensorflow_datasets)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para o gerador de números aleatórios (para reprodutibilidade)"
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Seed aleatória definida como: {args.seed}")
    
    print(f"Iniciando extração de dataset")
    print(f"Fonte: {args.source}")
    print(f"Destino: {args.dest}")
    print(f"Quantidade: {args.count} imagens")
    print()
    
    copiar_imagens_aleatorias(
        source_dir=args.source,
        dest_dir=args.dest,
        num_imagens=args.count,
        base_path=args.base
    )
    
    print(f"\nProcesso concluído!")


if __name__ == "__main__":
    main()
