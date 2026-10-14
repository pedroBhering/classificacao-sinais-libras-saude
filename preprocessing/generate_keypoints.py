import cv2
import numpy as np
import os
import mediapipe as mp
from tqdm import tqdm

mp_holistic = mp.solutions.holistic #Modelo Holistic do MediaPipe

def mediapipe_detection(image, model):
    #Converte o frame de BGR para RGB e retorna os resultados da detecção
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return model.process(image)

def extract_keypoints(results):
    #Junta pose (33 pontos), mão esquerda e mão direita (21 pontos cada) em um vetor de 225 valores (x, y, z)
    #Se alguma parte não for detectada, ela é preenchida com zeros
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def process_dataset(frames_root, keypoints_root, classes):
    #Extrai os keypoints de cada frame e salva um .npy por frame em keypoints_root/classe/video
    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9) as holistic:

        for classe in classes:
            classe_dir = os.path.join(frames_root, classe)
            if not os.path.isdir(classe_dir):
                continue

            for video_name in os.listdir(classe_dir):
                video_folder = os.path.join(classe_dir, video_name)
                if not os.path.isdir(video_folder):
                    continue

                frame_files = sorted(
                    [f for f in os.listdir(video_folder) if f.lower().endswith('.jpg')],
                    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
                )

                #Pula vídeos já processados. Para refazer um vídeo, apague a pasta dele em keypoints_root
                kp_folder = os.path.join(keypoints_root, classe, video_name)
                if os.path.isdir(kp_folder) and len(os.listdir(kp_folder)) == len(frame_files):
                    print(f"[AVISO] {classe}/{video_name} já processado. Pulando...")
                    continue
                os.makedirs(kp_folder, exist_ok=True)

                for frame_file in tqdm(frame_files, desc=f"{classe}/{video_name}", unit="frame"):
                    frame_path = os.path.join(video_folder, frame_file)
                    image = cv2.imread(frame_path)
                    if image is None:
                        continue

                    results = mediapipe_detection(image, holistic)
                    keypoints = extract_keypoints(results)

                    kp_filename = os.path.splitext(frame_file)[0] + '.npy'
                    np.save(os.path.join(kp_folder, kp_filename), keypoints)

    print("Processamento concluído.")

classes = ['Alergia', 'Autismo', 'Bronquite', 'Caimbra', 'Convulsao', 'Diarreia', 'Fratura', 'Frieira', 'Infarto', 'Sarampo']

frames_root = './BSL10_frames_sorted'
keypoints_root = './keypoints_data'

#Pegando os frames do conjunto de dados que passaram por transformações (data augmentation)
#frames_root = './BSL10_frames_augmented'
#keypoints_root = './keypoints_data_augmented'

#Para gerar os keypoints do especialista de alergia e frieira (todos os frames de cada vídeo, sem sorteio)
#Versões com transformações (20 por vídeo, geradas pelo frames_augmentation.py)
#classes = ['Alergia', 'Frieira']
#frames_root = './BSL10_ale_fri_aug'
#keypoints_root = './keypoints_data_ale_fri'

#Vídeos originais, sem transformações
#classes = ['Alergia', 'Frieira']
#frames_root = './BSL10_frames'
#keypoints_root = './keypoints_data_ale_fri_original'

process_dataset(frames_root, keypoints_root, classes)