import os
import re
import numpy as np
import shutil

def get_video_dirs(root_dir, class_names):
    #Retorna uma lista de (classe, nome do vídeo, caminho da pasta) para cada vídeo encontrado
    video_infos = []
    for cls in class_names:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            print(f"Atenção: diretório de classe não encontrado: {class_dir}")
            continue
        for entry in os.listdir(class_dir):
            video_path = os.path.join(class_dir, entry)
            if os.path.isdir(video_path):
                video_infos.append((cls, entry, video_path))
    return video_infos

def natural_sort_key(fname: str):
    #Ordenação correta para números de texto, então frame10.jpg vem depois de frame2.jpg
    parts = re.split(r'(\d+)', fname)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def sample_frames_normal(video_dir, target_count):
    #Sorteia target_count frames do vídeo, com mais chance para os frames do meio (distribuição normal)
    files = sorted(
        (f for f in os.listdir(video_dir) if f.endswith('.jpg')),
        key=natural_sort_key
    )
    N = len(files)
    if N < target_count:
        raise ValueError(f"Vídeo {video_dir} tem apenas {N} frames, necessário ≥ {target_count}")

    mu = (N - 1) / 2.0 #Centro da distribuição: frame do meio
    sigma = mu * 0.4 #Desvio padrão: 40% da distância até o centro

    idxs = np.arange(N)
    pdf = np.exp(-0.5 * ((idxs - mu) / sigma)**2)
    probs = pdf / pdf.sum()

    chosen = np.random.choice(idxs, size=target_count, replace=False, p=probs)
    chosen.sort()
    return chosen, files

def augment_videos(root_dir, new_dir, class_names, target_count, num_augment):
    #Para cada vídeo, gera num_augment versões, cada uma com target_count frames sorteados
    video_infos = get_video_dirs(root_dir, class_names)

    for cls, vid_name, vid_path in video_infos:
        class_out_dir = os.path.join(new_dir, cls)
        os.makedirs(class_out_dir, exist_ok=True)

        for aug_i in range(1, num_augment + 1):
            chosen_idxs, files = sample_frames_normal(vid_path, target_count)

            #Pasta de saída desta versão
            aug_dir = os.path.join(class_out_dir, f"{vid_name}_{aug_i}")
            os.makedirs(aug_dir, exist_ok=True)

            #Largura do zero à esquerda no nome dos frames
            pad = len(str(target_count - 1))
            for new_i, old_i in enumerate(chosen_idxs):
                src = os.path.join(vid_path, files[old_i])
                dst_fname = f"{new_i:0{pad}d}.jpg"
                dst = os.path.join(aug_dir, dst_fname)
                shutil.copy(src, dst)

            print(f"[{cls}/{vid_name}_{aug_i}] frames: {chosen_idxs.tolist()}")

    print("Augmentation concluída com sucesso.")

class_names = ['Alergia', 'Autismo', 'Bronquite', 'Caimbra', 'Convulsao','Diarreia', 'Fratura', 'Frieira', 'Infarto', 'Sarampo']

root_dir = './BSL10_frames'
new_dir = './BSL10_frames_sorted'
target_count = 30 #Quantidade de frames sorteados por versão
num_augment = 20 #Quantidade de versões geradas por vídeo

augment_videos(root_dir, new_dir, class_names, target_count, num_augment)