import json
import numpy as np
import tensorflow as tf
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

print("Происходит загрузка приложения. Подождите немного...")

pipe = pipeline("feature-extraction", model="DeepPavlov/rubert-base-cased")

def load_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["questions"], data["answers"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка загрузки данных: {e}")
        return [], []


def get_answer(question):
    question_embedding = np.mean(pipe(question)[0], axis=0)
    similarities = cosine_similarity([question_embedding], question_embeddings)[0]
    best_match_index = np.argmax(similarities) 

    similarity_threshold = 0.7

    if similarities[best_match_index] >= similarity_threshold:
      return answers[best_match_index]
    else:
      return "Извините, я не знаю ответа на ваш вопрос."

questions, answers = load_data("university_qa.json")

question_embeddings = np.load("question_embeddings.npy", allow_pickle=True)
answer_embeddings = np.load("answer_embeddings.npy", allow_pickle=True)

while True:
    question = input("Какой вопрос вас интересует?: ")
    response = get_answer(question)

    print(f"Ответ: {response}")
