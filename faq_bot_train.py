import os
import json
import numpy as np
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = TFAutoModel.from_pretrained("DeepPavlov/rubert-base-cased", from_pt=True)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]


def load_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["questions"], data["answers"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка загрузки данных: {e}")
        return [], []

questions, answers = load_data("university_qa.json")

question_embeddings = np.array([get_embedding(q) for q in questions])
answer_embeddings = np.array([get_embedding(a) for a in answers])


model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(question_embeddings.shape[1],)))
model.add(tf.keras.layers.Dense(256, activation='selu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
es = tf.keras.callbacks.EarlyStopping(monitor='auc', mode='max', patience=10, restore_best_weights=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])

model.compile(optimizer='adam', loss='cosine_similarity', metrics=['cosine_similarity'])

model.fit(question_embeddings, answer_embeddings, epochs=120)

model.save('FAQ.keras')
