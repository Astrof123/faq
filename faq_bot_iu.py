import os
import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import random

print("Происходит загрузка приложения. Подождите немного...")

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = TFAutoModel.from_pretrained("DeepPavlov/rubert-base-cased", from_pt=True)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

def get_answer(question, question_embeddings, answers, model):
    q_emb = get_embedding(question)
    similarities = np.dot(q_emb, answer_embeddings.T)
    best_match = np.argmax(similarities)
    return answers[best_match] if 0 <= best_match < len(answers) else "Извини, я не знаю ответа на этот вопрос."

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

loaded_model = tf.keras.models.load_model('FAQ.keras')


while True:
    user_question = input("Какой вопрос вас интересует?: ")
    response = get_answer(user_question, questions, answers, loaded_model)
    random_index = random.randint(0, len(response) - 1)
    
    print(f"Ответ: {response[random_index]}")
