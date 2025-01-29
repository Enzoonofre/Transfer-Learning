# Transfer Learning

## 📌 O que é Transfer Learning?

Transfer Learning, ou Aprendizado por Transferência, é uma técnica que reaproveita um modelo de aprendizado de máquina já treinado em um grande conjunto de dados para resolver um novo problema. Isso reduz drasticamente o tempo de treinamento e a necessidade de uma grande quantidade de dados, permitindo obter bons resultados com menos esforço computacional.

## 🚀 Como funciona?

A ideia é simples! Em vez de treinar um modelo do zero, seguimos estas etapas:

1. **Carregar um modelo base** com pesos pré-treinados.
2. **Congelar suas camadas**, para manter o conhecimento aprendido.
3. **Adicionar novas camadas** personalizadas para o seu problema.
4. **Treinar apenas as novas camadas** com seus próprios dados.

Dessa forma, aproveitamos o que já foi aprendido e ajustamos o modelo para nossa necessidade específica.

---

## 🔹 Passo 1: Carregar um Modelo Pré-Treinado

Ferramentas como o `Keras` já disponibilizam modelos prontos. Aqui está um exemplo usando o MobileNetV2:

```python
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf

pretrained_model = tf.keras.applications.MobileNetV2(
    include_top=False,  # Remove a última camada de classificação
    weights='imagenet'  # Usa pesos treinados na base ImageNet
)
```

🔹 `weights='imagenet'` → O modelo usará pesos pré-treinados na base ImageNet.
🔹 `include_top=False` → Removemos a última camada para personalizar a saída.

---

## 🔹 Passo 2: Congelar as Camadas do Modelo

Congelar as camadas significa que os pesos do modelo base não serão alterados durante o treinamento. Isso impede que o modelo "desaprenda" o que já sabe.

```python
pretrained_model.trainable = False
```

Assim, as camadas pré-treinadas continuam intactas e apenas as novas camadas serão ajustadas.

---

## 🔹 Passo 3: Criar um Novo Modelo

Agora adicionamos novas camadas para adaptar o modelo ao nosso problema específico:

```python
from tensorflow.keras import models, layers

model = models.Sequential([
    pretrained_model,  # Modelo pré-treinado
    layers.GlobalAveragePooling2D(),  # Camada de Pooling
    layers.Dense(2, activation='softmax')  # Camada final para classificação em 2 classes
])
```

Aqui, adicionamos uma camada de pooling global e uma camada densa final com duas saídas (para um problema de classificação binária).

---

## 🔹 Passo 4: Compilar e Treinar o Modelo

Agora, basta compilar e treinar o modelo com seus novos dados:

```python
import tensorflow.keras as keras

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()]
)

model.fit(new_dataset, epochs=20, callbacks=[...], validation_data=...)
```

Isso ajustará apenas as novas camadas adicionadas, mantendo o aprendizado do modelo base.

---

## 🎯 Conclusão

Transfer Learning é uma maneira poderosa de economizar tempo e recursos computacionais, utilizando modelos prontos e adaptando-os para novos problemas. Ao seguir esse processo, você pode criar modelos eficientes sem precisar treinar tudo do zero.

### 🔗 Próximos Passos:
✅ Testar diferentes modelos pré-treinados (ResNet, VGG, Inception, etc.)
✅ Ajustar hiperparâmetros para melhorar o desempenho
✅ Explorar técnicas de *fine-tuning* para liberar algumas camadas e refiná-las

---

📌 **Autor:** Enzo Onofre 
📌 **GitHub:** https://github.com/Enzoonofre
📌 **Contato:** enzo.onofre@ufu.br  

🚀 **Agora é sua vez!** Teste o Transfer Learning e veja como ele pode facilitar o desenvolvimento de modelos de aprendizado profundo. 😃

