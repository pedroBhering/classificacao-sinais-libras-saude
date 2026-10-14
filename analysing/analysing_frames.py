import os
import cv2
import mediapipe as mp
import svgwrite
import cairosvg

video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Alergia_e0_ALE_e0_02.mp4") 
output_svg = "mp_landmarks.svg"
output_pdf = output_svg.replace(".svg", ".pdf")

face_dx = -100 #Define o deslocamento horizontal do rosto em pixels (negativo = esquerda, positivo = direita)

margin = 20  #Margem (px) ao redor dos keypoints no recorte final


mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
mid_idx = total_frames // 2 #Pega o frame do meio do vídeo

cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Não foi possível ler o frame do vídeo.")

height, width = frame.shape[:2]


def draw_landmarks_svg(dwg, landmarks, connections, circle_color, line_color, circle_radius=2, line_width=1):
    #Desenha as conexões (linhas) e os pontos de um conjunto de landmarks no SVG
    if not landmarks:
        return

    if connections:
        for start_idx, end_idx in connections:
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]
            x1, y1 = int(start.x * width), int(start.y * height)
            x2, y2 = int(end.x * width), int(end.y * height)
            dwg.add(dwg.line(start=(x1, y1), end=(x2, y2),
                             stroke=line_color, stroke_width=line_width))

    for lm in landmarks.landmark:
        cx, cy = int(lm.x * width), int(lm.y * height)
        dwg.add(dwg.circle(center=(cx, cy), r=circle_radius, fill=circle_color))


def get_landmarks_bbox(results, width, height, margin=20):
    #Retorna (x, y, largura, altura) do retângulo que envolve todos os landmarks
    xs, ys = [], []

    def collect(landmarks):
        if landmarks:
            for lm in landmarks.landmark:
                xs.append(lm.x * width)
                ys.append(lm.y * height)

    collect(results.pose_landmarks)
    collect(results.face_landmarks)
    collect(results.left_hand_landmarks)
    collect(results.right_hand_landmarks)

    if not xs or not ys:
        return 0, 0, width, height

    min_x = int(min(xs)) - margin             
    min_y = max(int(min(ys)) - margin, 0)
    max_x = min(int(max(xs)) + margin, width)
    max_y = min(int(max(ys)) + margin, height)
    return min_x, min_y, max_x - min_x, max_y - min_y


with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    results = mediapipe_detection(frame, holistic)

    dwg = svgwrite.Drawing(output_svg, size=(width, height), profile='tiny')

    #Pose (corpo): pontos vermelhos, linhas pretas
    draw_landmarks_svg(
        dwg,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        circle_color="rgb(255,0,0)", #vermelho
        line_color="rgb(0,0,0)", #preto
        circle_radius=3,
        line_width=2
    )

    #Rosto: pontos verde-escuro, linhas verde-claro.
    #Para voltar à posição original, coloque face_dx = 0.
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            lm.x = (lm.x * width + face_dx) / width
        draw_landmarks_svg(
            dwg,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            circle_color="rgb(80,110,10)", #verde-escuro
            line_color="rgb(80,255,121)", #verde-claro
            circle_radius=1,
            line_width=1
        )

    #Mão esquerda: pontos vermelhos, linhas pretas
    draw_landmarks_svg(
        dwg,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        circle_color="rgb(255,0,0)", #vermelho
        line_color="rgb(0,0,0)", #preto
        circle_radius=3,
        line_width=2
    )

    #Mão direita: pontos vermelhos, linhas pretas
    draw_landmarks_svg(
        dwg,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        circle_color="rgb(255,0,0)", #vermelho
        line_color="rgb(0,0,0)", #preto
        circle_radius=3,
        line_width=2
    )

    #Recortando a área em branco ao redor dos keypoints
    min_x, min_y, crop_w, crop_h = get_landmarks_bbox(results, width, height, margin)
    dwg.viewbox(min_x, min_y, crop_w, crop_h)
    dwg['width'] = crop_w
    dwg['height'] = crop_h

    dwg.save()

#Exportando para pdf
cairosvg.svg2pdf(url=output_svg, write_to=output_pdf)
print(f"Frame do meio salvo como vetor em '{output_svg}' e '{output_pdf}'.")