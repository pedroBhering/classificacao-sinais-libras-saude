import os
import numpy as np
import matplotlib.pyplot as plt

def sample_frames_normal(total_frames, target_count):
    #Mesmo sorteio do aug_normal_frames.py (distribuição normal centrada no meio do vídeo), mas só com os índices
    if total_frames < target_count:
        raise ValueError(f"O vídeo tem apenas {total_frames} frames, necessário ≥ {target_count}")

    mu = (total_frames - 1) / 2.0 #Centro da distribuição: frame do meio
    sigma = mu * 0.4 #Desvio padrão: 40% da distância até o centro (20% do total de frames)

    idxs = np.arange(total_frames)
    pdf = np.exp(-0.5 * ((idxs - mu) / sigma)**2)
    probs = pdf / pdf.sum()

    chosen = np.random.choice(idxs, size=target_count, replace=False, p=probs)
    chosen.sort()
    return chosen

frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "BSL10_frames", "Alergia", "Alergia_AL_ALE_AL_01")
target_count = 30 #Quantidade de frames sorteados por sorteio
num_sorteios = 20 #Quantidade de sorteios repetidos sobre o mesmo vídeo
output_pdf = "distribuicao_sorteios_frames.pdf"
usar_latex = True #Fontes LaTeX no gráfico. Se der erro ou não quiser coloque False

plt.rcParams.update({
    "text.usetex": usar_latex,
    "font.family": "serif"
})

total_frames = len([f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')])
print(f"Total de frames do vídeo: {total_frames}")

all_choices = []
for i in range(num_sorteios):
    all_choices.append(sample_frames_normal(total_frames, target_count))

print("Primeiro sorteio de índices:")
print(all_choices[0])

all_idxs = np.concatenate(all_choices)

plt.figure(figsize=(8, 5))
plt.hist(all_idxs, bins=10, edgecolor='black')
plt.xlabel("Índice do quadro", fontsize=14)
plt.ylabel("Frequência de seleção", fontsize=14)
plt.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig(output_pdf)
plt.close()

print(f"\nGráfico salvo como '{output_pdf}'.")