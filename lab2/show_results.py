# -*- coding: utf-8 -*-
"""Печатает сохранённые результаты фактического запуска лабораторной работы."""
import json, argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
d = json.loads((BASE / "lab02_results.json").read_text(encoding="utf-8"))
p = argparse.ArgumentParser()
p.add_argument("--section", choices=["1", "2", "3"], default="1")
s = p.parse_args().section

def title(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)

if s == "1":
    r=d["task1"]; title("ЗАДАНИЕ 1. ТОКЕНИЗАЦИЯ")
    print("Исходный текст:",r["text"])
    print("sent_tokenize:",r["sentences"])
    print("word_tokenize:",r["words"])
    print("WordPunctTokenizer:",r["wordpunct"])
    print("Анализ:",r["analysis"])
    r=d["task2"]; title("ЗАДАНИЕ 2. СТЕММИНГ")
    print("Слова:",r["words"])
    for k,v in r["stemmers"].items(): print(k+":",v)
    print("Анализ:",r["analysis"])
    r=d["task3"]; title("ЗАДАНИЕ 3. ЛЕММАТИЗАЦИЯ")
    print("Существительные:",r["nouns"])
    print("Глаголы:",r["verbs"])
    print("Анализ:",r["analysis"])
    r=d["task4"]; title("ЗАДАНИЕ 4. ЧАНКИНГ BROWN")
    print(f"Слов: {r['source_words']}; размер чанка: {r['chunk_size']}; чанков: {r['chunk_count']}")
    for i,x in enumerate(r["chunks"],1): print(f"Чанк {i:02}: {x[:50]}")
    r=d["task5"]; title("ЗАДАНИЕ 5. BAG OF WORDS")
    print("Размер матрицы:",r["shape"])
    print("Словарь:",r["terms"])
    for name,row in zip(r["documents"],r["matrix"]): print(name, row)
elif s == "2":
    r=d["task6"]; title("ЗАДАНИЕ 6. TF-IDF + NAIVE BAYES")
    print("Категории:",r["categories"],"| документов:",r["train_size"])
    for x in r["predictions"]: print(x["predicted_category"],"|",x["text"])
    r=d["task7"]; title("ЗАДАНИЕ 7. ГРАММАТИЧЕСКИЙ РОД ИМЁН")
    for n,v in r["accuracy_by_n"].items(): print(f"N={n}: accuracy={v:.4f}")
    print("Лучшее N:",r["best_n"])
    for n,items in r["predictions"].items():
        print("N="+n, "; ".join(x["name"]+" → "+x["prediction"] for x in items))
    r=d["task8"]; title("ЗАДАНИЕ 8. СЕНТИМЕНТ-АНАЛИЗ")
    print(f"Обучение: {r['train_size']}; тест: {r['test_size']}; accuracy={r['accuracy']:.4f}")
    print(r["informative_features"])
    for x in r["samples"]: print(x["prediction"],x["probabilities"],"|",x["text"])
    r=d["task9"]; title("ЗАДАНИЕ 9. LDA")
    print("Документов:",r["documents"],"| словарь:",r["vocabulary_size"])
    for t in r["topics"]: print("Тема",t["topic"]+1,[(x["word"],round(x["weight"],3)) for x in t["words"]])
elif s == "3":
    r=d["task10"]; title("ЗАДАНИЕ 10. FINE-TUNING BERT-TINY")
    print("Задача:",r["task"])
    print("Датасет:",r["dataset"],"| модель:",r["model"])
    print("Примеры обучения:",r["train_examples"],"| validation:",r["validation_examples"],"| эпох:",r["epochs"])
    print("Метрики validation:",{k:round(v,4) for k,v in r["metrics"].items() if k in ["validation_accuracy","validation_f1","validation_loss"]})
    print("Метрики test:",{k:round(v,4) for k,v in r["metrics"].items() if k in ["test_accuracy","test_f1","test_loss"]})
    print("Описание датасета:",r.get("dataset_description"))
    print("Описание модели:",r.get("model_description"))
    for x in r["predictions"]:
        print(x["prediction"],{k:round(v,3) for k,v in x["probabilities"].items()},"|",x["text"])
    print("Модель сохранена в artifacts/bert_tiny_sst2")