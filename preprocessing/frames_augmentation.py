import os
import cv2
import random
from tqdm import tqdm

def process_dataset_with_augmentation(frames_root, augmented_root, classes, num_augmentations=1):
    #Gera num_augmentations versões de cada pasta de frames, aplicando rotação, translação e corte
    #Com 1 versão, a pasta de saída mantém o mesmo nome da entrada. Com mais, recebe o sufixo _01, _02...
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
            if not frame_files:
                continue

            print(f"\n{classe}/{video_name} — Gerando {num_augmentations} versões...")

            for aug_idx in range(num_augmentations):
                out_name = video_name if num_augmentations == 1 else f"{video_name}_{aug_idx+1:02d}"
                out_folder = os.path.join(augmented_root, classe, out_name)
                os.makedirs(out_folder, exist_ok=True)

                #Parâmetros sorteados uma vez por versão, iguais para todos os frames dela
                angle = random.uniform(-10, 10) #Rotação em graus
                tx = random.uniform(-20, 20) #Translação em x em pixels
                ty = random.uniform(-20, 20) #Translação em y em pixels
                p = random.uniform(0, 0.10) #Fração cortada em cada borda

                for frame_file in tqdm(frame_files, desc=out_name, leave=False):
                    img_path = os.path.join(video_folder, frame_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    h, w = img.shape[:2]

                    #Rotação e translação
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    M[0, 2] += tx
                    M[1, 2] += ty
                    warped = cv2.warpAffine(
                        img, M, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT
                    )

                    #Corte centralizado e retorno ao tamanho original
                    crop_x = int(w * p)
                    crop_y = int(h * p)
                    cropped = warped[crop_y:h-crop_y, crop_x:w-crop_x]
                    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

                    out_path = os.path.join(out_folder, frame_file)
                    cv2.imwrite(out_path, resized)

    print("\nFim da geração de augmentations!")

classes = ['Alergia', 'Autismo', 'Bronquite', 'Caimbra', 'Convulsao', 'Diarreia', 'Fratura', 'Frieira', 'Infarto', 'Sarampo']

frames_root = './BSL10_frames_sorted'
augmented_root = './BSL10_frames_augmented'
num_augmentations = 1

#Para gerar as versões do especialista de alergia e frieira (frames originais, 20 versões por vídeo)
#classes = ['Alergia', 'Frieira']
#frames_root = './BSL10_frames'
#augmented_root = './BSL10_ale_fri_aug'
#num_augmentations = 20

process_dataset_with_augmentation(frames_root, augmented_root, classes, num_augmentations)