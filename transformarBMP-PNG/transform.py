import os
from PIL import Image
import sys

def convert_bmp_to_png():
    """
    Converte todas as imagens BMP da pasta imagesBMP para PNG na pasta imagesPNG
    """
    # Definir os caminhos das pastas
    input_folder = "imagesBMP"
    output_folder = "imagesPNG"
    
    # Verificar se a pasta de entrada existe
    if not os.path.exists(input_folder):
        print(f"Erro: A pasta '{input_folder}' não existe.")
        return
    
    # Criar pasta de saída se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Pasta '{output_folder}' criada.")
    
    # Listar todos os arquivos BMP na pasta de entrada
    bmp_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]
    
    if not bmp_files:
        print(f"Nenhum arquivo BMP encontrado na pasta '{input_folder}'.")
        return
    
    print(f"Encontrados {len(bmp_files)} arquivos BMP para conversão.")
    
    converted_count = 0
    error_count = 0
    
    # Converter cada arquivo BMP para PNG
    for bmp_file in bmp_files:
        try:
            # Caminhos completos dos arquivos
            input_path = os.path.join(input_folder, bmp_file)
            output_filename = os.path.splitext(bmp_file)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            
            # Abrir e converter a imagem
            with Image.open(input_path) as img:
                # Converter para RGB se necessário (BMP pode estar em outros modos)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Salvar como PNG
                img.save(output_path, 'PNG')
                
            print(f"✓ Convertido: {bmp_file} → {output_filename}")
            converted_count += 1
            
        except Exception as e:
            print(f"✗ Erro ao converter {bmp_file}: {str(e)}")
            error_count += 1
    
    # Relatório final
    print(f"\n--- Relatório de Conversão ---")
    print(f"Arquivos convertidos com sucesso: {converted_count}")
    print(f"Arquivos com erro: {error_count}")
    print(f"Total de arquivos processados: {len(bmp_files)}")

if __name__ == "__main__":
    try:
        convert_bmp_to_png()
    except KeyboardInterrupt:
        print("\n\nOperação cancelada pelo usuário.")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
        sys.exit(1)