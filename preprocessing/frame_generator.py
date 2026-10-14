import os
import cv2
from tqdm import tqdm

classes = ['Alergia', 'Autismo', 'Bronquite', 'Caimbra', 'Convulsao', 'Diarreia', 'Fratura', 'Frieira', 'Infarto', 'Sarampo']

#Pastas de leitura dos vídeos e salvamento dos frames
base_dir = './BSL10'
output_base_dir = './BSL10_frames'


TARGET_WIDTH = 640 #Largura final de cada frame em pixels
TARGET_HEIGHT = 480 #Altura final de cada frame em pixels

for classe in classes:
    input_dir = os.path.join(base_dir, classe)
    if not os.path.isdir(input_dir):
        print(f"Pasta de vídeos não encontrada: {input_dir}")
        continue

    videos = [f for f in os.listdir(input_dir)
              if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for video in videos:
        video_path = os.path.join(input_dir, video)
        video_name = os.path.splitext(video)[0]

        #Cria a pasta de saída: output_base_dir/classe/video_name
        frames_dir = os.path.join(output_base_dir, classe, video_name)
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Não foi possível abrir o vídeo: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Extraindo {total_frames} frames de {video}")

        for idx in tqdm(range(total_frames), desc=video_name, unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            frame_filename = f"frame_{idx}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame_resized)

        cap.release()
        print(f"Concluído: {video_name}. Frames salvos em {frames_dir}\n")