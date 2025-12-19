import os
from PIL import Image
from tqdm import tqdm # type: ignore # pip install tqdm

def convert_and_resize(input_folder, output_folder, target_size=(1024, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Filtra apenas arquivos BMP
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]
    
    print(f"Encontrados {len(files)} arquivos BMP. Convertendo para PNG...")

    for filename in tqdm(files):
        try:
            input_path = os.path.join(input_folder, filename)
            
            # Define o nome de saída trocando a extensão para .png
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = name_without_ext + ".png"
            output_path = os.path.join(output_folder, output_filename)

            with Image.open(input_path) as img:
                # Converte para RGB (caso algum BMP esteja em modo diferente)
                img = img.convert("RGB")

                # Redimensiona usando LANCZOS (melhor preservação de detalhes)
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Salva como PNG
                # optimize=True ajuda a reduzir o tamanho do arquivo sem perda
                img_resized.save(output_path, format='PNG', optimize=True)

        except Exception as e:
            print(f"Erro no arquivo {filename}: {e}")

    print("Conversão e resize finalizados!")

# --- Configuração ---
diretorio_origem = './CTC-360'
diretorio_destino = './CTC-360-resized'

convert_and_resize(diretorio_origem, diretorio_destino)