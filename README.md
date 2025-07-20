# sist-rec-facial
Projeto DIO  Criando um Sistema de Reconhecimento Facial do Zero

Ok, vamos criar um README para guiar a criação de um sistema de reconhecimento facial do zero usando YOLO e Colab.

## Resultados

![Uploading image.png…]()

<img width="1136" height="723" alt="image" src="https://github.com/user-attachments/assets/c24a6d22-84f4-45ae-bcc0-220daa6deb72" />


```markdown
# Criando um Sistema de Reconhecimento Facial do Zero no Colab com YOLO

Este guia descreve como construir um sistema de reconhecimento facial simples usando YOLO para detecção e uma rede de classificação (que você precisará treinar) para identificar os rostos detectados.

## Pré-requisitos

*   Conta Google (para usar o Google Colab)
*   Conhecimento básico de Python e TensorFlow

## Passo a Passo

1.  **Abrir um Notebook no Google Colab:**
    *   Acesse o Google Colab ([https://colab.research.google.com/](https://colab.research.google.com/)).
    *   Crie um novo notebook (File > New Notebook).

2.  **Instalar Dependências:**
    *   Execute as seguintes células no Colab para instalar as bibliotecas necessárias:

    ```python
    !pip install opencv-python
    !pip install tensorflow
    !pip install pillow
    # Outras bibliotecas que você possa precisar
    ```

3.  **Importar as Bibliotecas:**

    ```python
    import cv2
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    ```

4.  **Implementar a Detecção de Rosto com YOLO:**

    *   **Baixar os Pesos e a Configuração do YOLO:**
        *   Você precisará dos arquivos de configuração e pesos pré-treinados do YOLO.  Você pode utilizar modelos já treinados para detecção de rostos, ou utilizar o YOLO tradicional e fazer um fine tuning, para melhor performance.  Você pode procurar por modelos YOLO pré-treinados para detecção de rostos na internet.  Por exemplo:

            ```python
            # Exemplo de como baixar os arquivos (substitua pelos URLs corretos)
            !wget https://exemplo.com/yolov3-face.cfg -O yolov3-face.cfg
            !wget https://exemplo.com/yolov3-face.weights -O yolov3-face.weights
            ```

    *   **Carregar o Modelo YOLO:**

        ```python
        net = cv2.dnn.readNet("yolov3-face.weights", "yolov3-face.cfg")  # Substitua pelos nomes corretos

        # Carregar as classes (se aplicável ao seu modelo)
        classes = []  # Ex:  ['face']
        # with open("coco.names", "r") as f:  # Ou seu arquivo de classes
        #     classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        ```

    *   **Função para Detectar Rostos:**

        ```python
        def detect_faces(image):
            height, width, channels = image.shape

            # Preprocessar a imagem para o YOLO
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)  # ajuste o tamanho conforme necessário
            net.setInput(blob)
            outs = net.forward(output_layers)

            boxes = []
            confidences = []

            for out in outs:
                for detection in out:
                    confidence = detection[5]  # Ajuste o índice conforme necessário (pode variar entre modelos)
                    if confidence > 0.5:  # Limiar de confiança
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            return boxes, confidences
        ```

    *   **Aplicar a Detecção e Desenhar os Retângulos:**

        ```python
        def draw_boxes(image, boxes, confidences):
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"Conf: {confidences[i]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image
        ```

5.  **Implementar a Classificação de Rostos:**

    *   **Coletar e Preparar um Dataset:**
        *   Você precisará de um dataset de imagens de rostos das pessoas que você deseja reconhecer.  Organize as imagens em pastas separadas para cada pessoa.
    *   **Criar um Modelo de Classificação:**
        *   Use TensorFlow/Keras para criar um modelo CNN para classificar os rostos.  Um exemplo simples:

            ```python
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Ajuste o input_shape conforme necessário
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(num_classes, activation='softmax')  # num_classes = número de pessoas
            ])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Carregue e prepare os dados
            # ... (código para carregar imagens, redimensionar para 128x128, normalizar, etc.)

            # Treinar o modelo
            # model.fit(x_train, y_train, epochs=10) # Ajuste os epochs

            ```

    *   **Função para Classificar o Rosto Detectado:**

        ```python
        def classify_face(face_image):
            # Pré-processar a imagem para o modelo de classificação
            resized_face = cv2.resize(face_image, (128, 128)) # Ajuste o tamanho
            img_array = tf.keras.utils.img_to_array(resized_face)
            img_array = tf.expand_dims(img_array, 0) # Adiciona uma dimensão para o batch
            img_array /= 255.0  # Normalizar

            # Fazer a previsão
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            return predicted_class, confidence # Retorna o nome da classe e a confiança

        ```
6.  **Integrar Detecção e Classificação:**
    *   Modifique o loop de detecção para recortar os rostos detectados e classificá-los.
    ```python
        # Dentro da função draw_boxes, depois de detectar o rosto:
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            face_roi = image[y:y+h, x:x+w]  # Recorta a região do rosto

            if face_roi.size != 0: # Garante que o rosto foi detectado
                predicted_class, confidence = classify_face(face_roi)

                # Obter o nome da classe (se você tiver um mapeamento de índice para nome)
                name = class_names[predicted_class]  # Exemplo: class_names = ['pessoa1', 'pessoa2', ...]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{name}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    ```
7.  **Testar:**
    *   Carregue uma imagem para testar.

    ```python
    image_path = "path/para/sua/imagem.jpg"  # Substitua pelo caminho da sua imagem
    image = cv2.imread(image_path)

    boxes, confidences = detect_faces(image)
    image = draw_boxes(image, boxes, confidences) # Esta função agora também classifica

    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

## Sugestões de Melhorias

*   **Aumento de Dados:**  Aumentar o tamanho do seu dataset de rostos melhorará a precisão da classificação.  Considere usar técnicas de aumento de dados (rotação, zoom, etc.).
*   **Arquitetura do Modelo:** Experimentar diferentes arquiteturas de modelos de classificação (por exemplo, ResNet, MobileNet) pode levar a melhores resultados.
*   **Ajuste Fino do YOLO:**  Fazer um "fine-tuning" do YOLO para detectar rostos especificamente pode melhorar a precisão da detecção.
*   **TensorFlow Lite:** Converter o modelo de classificação para TensorFlow Lite para melhor desempenho e otimização para dispositivos móveis.
*   **Embeddings Faciais:** Considerar usar técnicas como FaceNet ou OpenFace para gerar embeddings faciais e usar um classificador simples (como SVM) para identificar os rostos. Isso geralmente produz resultados mais robustos.

## Observações

*   Adapte os caminhos dos arquivos, parâmetros e arquiteturas dos modelos de acordo com sua necessidade.
*   A detecção de rostos com YOLO e a classificação são tarefas complexas. A precisão dependerá da qualidade do dataset e da configuração dos modelos.
*   Lembre-se de ajustar os tamanhos de imagens (`416x416` para YOLO e `128x128` para a classificação) conforme necessário.
```

**Explicação e Melhorias Adicionais:**

*   **YOLO:** O YOLO (You Only Look Once) é um detector de objetos rápido. Neste caso, usamos para detectar onde estão os rostos na imagem.
*   **Coordenadas do Retângulo:** O YOLO fornece as coordenadas (x, y, largura, altura) de um retângulo ao redor de cada rosto detectado. Você usará essas coordenadas para recortar a região do rosto da imagem original.
*   **Classificação:** Após recortar o rosto, você o passa para um modelo de classificação que você treinou para reconhecer diferentes pessoas.
*   **Colab:** O Google Colab é um ambiente gratuito de notebook Python executado na nuvem, ótimo para prototipagem e experimentação com aprendizado de máquina.
*   **Dataset:** Um bom dataset é fundamental para o sucesso do reconhecimento facial. Quanto mais imagens de cada pessoa você tiver, e quanto mais variadas forem essas imagens (diferentes ângulos, iluminação, expressões), melhor será o desempenho do seu sistema.
*   **Aumento de Dados:** Técnicas como rotação, zoom, e pequenas transformações na imagem podem aumentar seu dataset de forma artificial, melhorando a robustez do modelo.
*   **Embeddings Faciais (FaceNet, OpenFace):** Em vez de treinar um classificador diretamente nas imagens dos rostos, essas técnicas aprendem a representar cada rosto como um vetor (embedding) de características. Rostos da mesma pessoa terão embeddings mais próximos no espaço vetorial. Isso torna o reconhecimento mais robusto a variações na aparência.

Este README fornece um ponto de partida sólido. Adapte-o, experimente e divirta-se construindo seu sistema de reconhecimento facial!
