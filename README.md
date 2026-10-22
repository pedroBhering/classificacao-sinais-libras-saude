# Classificação de sinais dinâmicos de Libras relacionados ao contexto hospitalar utilizando redes neurais recorrentes

O projeto é composto por códigos que realizam o treinamento de modelos para a classificação de 10 sinais de Libras da área da saúde (Alergia, Autismo, Bronquite, Câimbra, Convulsão, Diarreia, Fratura, Frieira, Infarto e Sarampo). Os pontos do corpo e das mãos de cada vídeo são extraídos com o MediaPipe Holistic, e uma rede LSTM classifica a sequência de 30 quadros (225 características por quadro).

Além do modelo de 10 classes, o projeto traz uma estratégia hierárquica: um modelo de 7 classes, em que os pares de sinais parecidos (Alergia e Frieira, Câimbra e Fratura, Bronquite e Infarto) viram uma classe só, e três modelos especialistas que separam cada par.

## Estrutura

```
preprocessing/   vídeos -> quadros -> sorteio e aumento de dados -> keypoints
analysing/       figuras da seção de pré-processamento (marcos do MediaPipe e sorteio dos quadros)
models/          notebooks de treino (validação cruzada e modelo final) e, em models/models/, os pesos treinados
evaluation/      notebooks de avaliação, um para cada notebook de treino
results/         matrizes de confusão geradas pelos notebooks de avaliação
requirements/    dependências dos dois ambientes
```

## Ambientes

São dois ambientes Python 3.10, um para o pré-processamento e outro para o treino e a avaliação:

```bash
python3.10 -m venv env_dataset
source env_dataset/bin/activate
pip install -r requirements/dataset.txt

python3.10 -m venv env_training
source env_training/bin/activate
pip install -r requirements/training.txt
```

As figuras e matrizes usam LaTeX nas fontes (`usar_latex = True`). Caso não tenha o LaTeX instalado pode colocar `usar_latex = False` nos notebooks e scripts.

Os experimentos do trabalho foram executados em CPU (Intel Core i5-10300H, 16 GB de RAM), no WSL 2 com Ubuntu 22.04.

## Dados

Os keypoints (marcos) de todos os quadros dos vídeos são disponibilizados separadamente, juntamente com vídeos de exemplo dos sinais feitos pelo autor do trabalho: https://drive.google.com/drive/folders/1oxThRrIKD-KX2nFlai2Zg_fl44xwPffg?usp=sharing. **Os vídeos originais não serão publicados.**

**Observação:** como o autor não é fluente em Libras, algumas execuções podem não corresponder exatamente à forma correta dos sinais. Os dados servem para reproduzir o pipeline e os experimentos e para demonstrar, de maneira aproximada, como os sinais são realizados, não como referência linguística de Libras.

Depois de baixar, as pastas devem ficar na raiz do repositório:

| Pasta | Conteúdo |
|---|---|
| `keypoints_data` | 20 versões de 30 quadros sorteados de cada vídeo (`<vídeo>_1` a `<vídeo>_20`) |
| `keypoints_data_augmented` | mesmos sorteios, com transformações nas imagens (rotação, translação e corte); usados no treino |
| `keypoints_data_ale_fri` | todos os quadros dos vídeos de Alergia e Frieira, com 20 versões transformadas de cada vídeo (treino do especialista) |
| `keypoints_data_ale_fri_original` | todos os quadros dos vídeos de Alergia e Frieira, sem transformações (validação do especialista) |
| `keypoints_data_test` | todos os quadros dos 60 vídeos do conjunto de teste adicional (6 por sinal) |

**Observação:** Caso deseje, é possível montar o seu próprio conjunto de dados e realizar o passo a passo de execução para obter os próprios resultados. 

## Como executar

### 1. Pré-processamento (opcional, se os keypoints já estiverem na raiz)

Execute da raiz do repositório, com o `env_dataset`. Cada script tem, no final, as pastas de entrada e saída, e as variações comentadas.

| Ordem | Script | Faz |
|---|---|---|
| 1 | `preprocessing/frame_generator.py` | extrai todos os quadros de cada vídeo |
| 2 | `preprocessing/aug_normal_frames.py` | sorteia 30 quadros por versão, com mais chance para os quadros do meio do vídeo (distribuição normal) |
| 3 | `preprocessing/frames_augmentation.py` | aplica rotação, translação e corte aos quadros |
| 4 | `preprocessing/generate_keypoints.py` | extrai os keypoints com o MediaPipe Holistic |

Para o especialista de Alergia e Frieira, o passo 2 é pulado: o passo 3 usa os quadros completos (`BSL10_frames`) com 20 versões, e o passo 4 gera `keypoints_data_ale_fri` e `keypoints_data_ale_fri_original` (as configurações estão comentadas nos scripts).

Em `analysing/` (com o `env_dataset`), `analysing_frames.py` gera a figura dos marcos do MediaPipe e `sort_norm.py` gera a figura do sorteio dos quadros.

### 2. Treino

Execute os notebooks de `models/` com o `env_training`. Cada um treina com validação cruzada leave-one-subject-out (10 folds) e depois, exceto o de 10 classes, com todos os dados. Os pesos são salvos em `models/models/<nome>/` (`model_fold_N.keras` e o modelo final). Os pesos treinados já estão incluídos no repositório, então é possível pular esta etapa e ir direto para a avaliação. O `ale_fri_model.ipynb` também detecta o início do movimento em cada vídeo antes do treino.

### 3. Avaliação

Execute os notebooks de `evaluation/` com o `env_training`. Os pesos são lidos de `models/models/` (os que acompanham o repositório ou os gerados no treino).

| Treino | Avaliação |
|---|---|
| `10_classes_model` | `10_classes_eval` |
| `7_classes_model` | `7_classes_eval` |
| `bro_inf_model` | `bro_inf_eval` |
| `cai_fra_model` | `cai_fra_eval` |
| `ale_fri_model` | `ale_fri_eval` |
| 7 classes e os três especialistas | `hierarchical_eval` |
| 7 classes e os três especialistas (conjunto de teste adicional) | `hierarchical_test_eval` |

O `ale_fri_eval` e o `hierarchical_eval` usam o conjunto de validação criado pelas células de preparação de dados do `ale_fri_model.ipynb`, que devem ser executadas antes (mesmo sem treinar). Os dois `hierarchical_*` dependem do modelo de 7 classes e dos três especialistas.

As matrizes de confusão são salvas em `results/`.

**Sobre o sorteio:** o `hierarchical_test_eval` sorteia 20 conjuntos novos do teste adicional a cada execução. Por isso os valores obtidos podem variar um pouco entre execuções.

**Sobre os resultados e os parâmetros:** os resultados gravados nos notebooks vêm de uma outra execução e podem diferir dos reportados no trabalho. Além disso, os parâmetros dos notebooks também podem diferir dos descritos no trabalho, pois aqui estão apenas para ilustrar as execuções.

## Citações

BITTENCOURT, Pedro de Araújo Bhering. *Classificação de sinais dinâmicos de Libras relacionados ao contexto hospitalar utilizando redes neurais recorrentes*. 2026. Trabalho de Conclusão de Curso (Bacharelado em Engenharia Computacional) – Universidade Federal de Juiz de Fora, Juiz de Fora, 2026.

de Araújo Bhering Bittencourt, P., da Silva Maciel, L.M., Vieira, M.B., Villela, S.M. (2027). Classification of Dynamic Libras Signs Related to the Hospital Context Using Recurrent Neural Networks. In: Gervasi, O., et al. Computational Science and Its Applications – ICCSA 2026. ICCSA 2026. Lecture Notes in Computer Science, vol 16767. Springer, Cham. https://doi.org/10.1007/978-3-032-30491-9_33
