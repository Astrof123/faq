import numpy as np
import tensorflow as tf
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

pipe = pipeline("feature-extraction", model="DeepPavlov/rubert-base-cased")

def load_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["questions"], data["answers"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка загрузки данных: {e}")
        return [], []

questions, answers = load_data("university_qa.json")

question_embeddings = [np.mean(pipe(q)[0], axis=0) for q in questions]
answer_embeddings = [np.mean(pipe(a)[0], axis=0) for a in answers]

np.save("question_embeddings.npy", question_embeddings)
np.save("answer_embeddings.npy", answer_embeddings)

def get_answer(question):
    question_embedding = np.mean(pipe(question)[0], axis=0)
    similarities = cosine_similarity([question_embedding], question_embeddings)[0]
    best_match_index = np.argmax(similarities) 

    similarity_threshold = 0.7

    if similarities[best_match_index] >= similarity_threshold:
      return answers[best_match_index]
    else:
      return "Извините, я не знаю ответа на ваш вопрос."


# Пример
print(get_answer("Как узнать результаты экзаменов?"))
