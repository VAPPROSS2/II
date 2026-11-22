# -*- coding: utf-8 -*-
"""
Лабораторная работа №2. Решение задач NLP.
Запуск: python lab02.py
Полный запуск включает fine-tuning DistilBERT (загрузка датасета/модели из Hugging Face).
Для локальной проверки заданий 1–9: python lab02.py --skip-transformer
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, traceback
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "lab02_data.txt"
ARTIFACTS = BASE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
RESULTS = {}

def save_results():
    (BASE / "lab02_results.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

def ensure_nltk():
    import nltk
    packages = ["punkt", "punkt_tab", "wordnet", "omw-1.4", "brown",
                "names", "movie_reviews", "stopwords"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as exc:
            print(f"Предупреждение: не удалось скачать {pkg}: {exc}")
    return nltk

def task1(nltk):
    section("ЗАДАНИЕ 1. ТОКЕНИЗАЦИЯ")
    from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
    text = (
        "NLP-based systems cannot always interpret state-of-the-art models correctly. "
        "It is used in search engines, chatbots, and document analysis! "
        "Context is important: the same word can have different meanings."
    )
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        punct = WordPunctTokenizer().tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
        sentences = sent_tokenize(text); words = word_tokenize(text)
        punct = WordPunctTokenizer().tokenize(text)
    print("Исходный текст:", text)
    print("sent_tokenize:", sentences)
    print("word_tokenize:", words)
    print("WordPunctTokenizer:", punct)
    analysis = ("sent_tokenize выделяет предложения; word_tokenize делит текст на слова "
                "и стандартные знаки препинания; WordPunctTokenizer отделяет пунктуацию "
                "максимально явно, включая дефисы и апострофы.")
    print("Анализ:", analysis)
    RESULTS["task1"] = {"text": text, "sentences": sentences, "words": words,
                        "wordpunct": punct, "analysis": analysis}

def task2():
    section("ЗАДАНИЕ 2. СТЕММИНГ")
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.snowball import SnowballStemmer
    words = ["sing", "sang", "singer", "singing", "calves", "ponies",
             "running", "studies", "relational", "happiness"]
    stemmers = {"Porter": PorterStemmer(), "Lancaster": LancasterStemmer(),
                "Snowball": SnowballStemmer("english")}
    result = {name: [stemmer.stem(w) for w in words] for name, stemmer in stemmers.items()}
    print("Исходные слова:", words)
    for name, stems in result.items(): print(f"{name}: {stems}")
    analysis = ("Стеммеры используют разные эвристики отсечения окончаний. Lancaster обычно "
                "агрессивнее и чаще сокращает слова сильнее; Porter и Snowball дают близкие, "
                "но не всегда одинаковые основы. Полученная основа не обязательно является словом.")
    print("Анализ:", analysis)
    RESULTS["task2"] = {"words": words, "stemmers": result, "analysis": analysis}

def task3(nltk):
    section("ЗАДАНИЕ 3. ЛЕММАТИЗАЦИЯ")
    from nltk.stem import WordNetLemmatizer
    words = RESULTS["task2"]["words"]
    lemmatizer = WordNetLemmatizer()
    try:
        nouns = [lemmatizer.lemmatize(w, pos="n") for w in words]
        verbs = [lemmatizer.lemmatize(w, pos="v") for w in words]
    except LookupError:
        nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
        nouns = [lemmatizer.lemmatize(w, pos="n") for w in words]
        verbs = [lemmatizer.lemmatize(w, pos="v") for w in words]
    print("Исходные слова:", words)
    print("Леммы (существительные):", nouns)
    print("Леммы (глаголы):", verbs)
    analysis = ("Лемматизация учитывает словарь и часть речи. Например, calves преобразуется "
                "в calf в режиме существительного, а running — в run в режиме глагола. "
                "Результат зависит от корректно заданной части речи.")
    print("Анализ:", analysis)
    RESULTS["task3"] = {"words": words, "nouns": nouns, "verbs": verbs, "analysis": analysis}

def task4(nltk):
    section("ЗАДАНИЕ 4. ЧАНКИНГ")
    from nltk.corpus import brown
    try: words = brown.words()[:1200]
    except LookupError:
        nltk.download("brown", quiet=True); words = brown.words()[:1200]
    n, m = len(words), 100
    chunks = [" ".join(words[i:i+m]) for i in range(0, n, m)]
    print(f"Из коллекции Brown взято {n} слов; размер чанка M={m}; всего чанков: {len(chunks)}")
    for i, chunk in enumerate(chunks[:20], 1): print(f"Чанк {i:02d}: {chunk[:50]}")
    RESULTS["task4"] = {"source_words": n, "chunk_size": m, "chunk_count": len(chunks),
                        "chunks": chunks[:20], "analysis": "Чанкинг объединяет фиксированное число слов в последовательные блоки; в отличие от токенизации, его цель — сформировать более крупные фрагменты текста."}
    return chunks

def task5(chunks):
    section("ЗАДАНИЕ 5. BAG OF WORDS")
    from sklearn.feature_extraction.text import CountVectorizer
    documents = {f"chunk_{i+1}": chunk for i, chunk in enumerate(chunks[:12])}
    vectorizer = CountVectorizer(max_features=20, stop_words="english")
    matrix = vectorizer.fit_transform(documents.values())
    terms = vectorizer.get_feature_names_out().tolist()
    dense = matrix.toarray().tolist()
    print("Документов:", matrix.shape[0], "| признаков:", matrix.shape[1])
    print("Словарь:", terms)
    print("Терм-документная матрица (строки — чанки, столбцы — слова):")
    print("       " + " ".join(f"{t:>9}" for t in terms))
    for name, row in zip(documents.keys(), dense):
        print(f"{name:>9} " + " ".join(f"{v:>9}" for v in row))
    analysis = ("Каждая строка соответствует чанку, каждый столбец — слову словаря. "
                "Значение ячейки показывает число вхождений слова в чанк. Порядок слов "
                "и грамматическая структура в Bag of Words не учитываются.")
    print("Анализ:", analysis)
    RESULTS["task5"] = {"shape": list(matrix.shape), "terms": terms, "matrix": dense,
                        "documents": list(documents.keys()), "analysis": analysis}

def task6():
    section("ЗАДАНИЕ 6. КЛАССИФИКАЦИЯ ТЕКСТОВ TF-IDF + NAIVE BAYES")
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    categories = ["comp.graphics", "rec.sport.baseball", "sci.space",
                  "talk.politics.misc", "rec.autos"]
    print("Категории:", categories)
    train = fetch_20newsgroups(subset="train", categories=categories,
                               remove=("headers", "footers", "quotes"))
    vectorizer = CountVectorizer(stop_words="english", max_features=25000,
                                 min_df=2, max_df=0.95)
    counts = vectorizer.fit_transform(train.data)
    tfidf = TfidfTransformer()
    train_tfidf = tfidf.fit_transform(counts)
    classifier = MultinomialNB().fit(train_tfidf, train.target)
    tests = [
        "The new graphics card renders realistic three dimensional images.",
        "The baseball team won the championship after a strong season.",
        "NASA scientists discovered a new planet orbiting a distant star.",
        "The senator presented a proposal during the national election debate.",
        "The electric car has a longer range and faster charging technology."
    ]
    test_counts = vectorizer.transform(tests)
    test_tfidf = tfidf.transform(test_counts)
    predictions = classifier.predict(test_tfidf)
    output = [{"text": text, "predicted_category": train.target_names[int(pred)]}
              for text, pred in zip(tests, predictions)]
    print("Размер обучающей выборки:", len(train.data))
    for item in output: print(f"- {item['predicted_category']}: {item['text']}")
    analysis = ("Классификатор MultinomialNB использует TF-IDF-векторы. Категории с "
                "характерной лексикой распознаются по статистическим признакам; короткие "
                "или неоднозначные предложения могут классифицироваться ошибочно.")
    print("Анализ:", analysis)
    RESULTS["task6"] = {"categories": categories, "train_size": len(train.data),
                        "predictions": output, "analysis": analysis}

def task7(nltk):
    section("ЗАДАНИЕ 7. ОПРЕДЕЛЕНИЕ ГРАММАТИЧЕСКОГО РОДА ИМЁН")
    from nltk.corpus import names
    from nltk import NaiveBayesClassifier
    from nltk.classify import accuracy as nltk_accuracy
    try:
        male = [(name, "male") for name in names.words("male.txt")]
        female = [(name, "female") for name in names.words("female.txt")]
    except LookupError:
        nltk.download("names", quiet=True)
        male = [(name, "male") for name in names.words("male.txt")]
        female = [(name, "female") for name in names.words("female.txt")]
    data = male + female
    random.Random(42).shuffle(data)
    def features(name, n):
        name = name.lower()
        return {"suffix": name[-n:]}
    custom_names = ["Alex", "Amelia", "Jordan", "Victoria", "Mark", "Taylor", "Sasha"]
    scores, predictions = {}, {}
    for n in range(1, 6):
        split = int(len(data) * 0.8)
        train = [(features(name,n), label) for name,label in data[:split]]
        test = [(features(name,n), label) for name,label in data[split:]]
        clf = NaiveBayesClassifier.train(train)
        score = nltk_accuracy(clf, test)
        scores[str(n)] = score
        predictions[str(n)] = [{"name": name, "prediction": clf.classify(features(name,n))}
                               for name in custom_names]
        print(f"N={n}: accuracy={score:.4f}; " +
              ", ".join(f"{p['name']}→{p['prediction']}" for p in predictions[str(n)]))
    best_n = max(scores, key=scores.get)
    analysis = (f"Наибольшая точность на отложенной выборке получена при N={best_n} "
                f"({scores[best_n]:.4f}). Эвристика использует только последние буквы имени, "
                "поэтому она не определяет биологический пол человека и может ошибаться "
                "на унисекс-именах и именах из других языковых традиций.")
    print("Оптимальное N:", best_n, "|", analysis)
    RESULTS["task7"] = {"accuracy_by_n": scores, "predictions": predictions,
                        "best_n": int(best_n), "analysis": analysis}

def task8(nltk):
    section("ЗАДАНИЕ 8. СЕНТИМЕНТ-АНАЛИЗ ОТЗЫВОВ")
    from nltk.corpus import movie_reviews
    from nltk.classify import NaiveBayesClassifier
    from nltk.classify.util import accuracy as nltk_accuracy
    try:
        pos_ids = movie_reviews.fileids("pos")
        neg_ids = movie_reviews.fileids("neg")
    except LookupError:
        nltk.download("movie_reviews", quiet=True)
        pos_ids = movie_reviews.fileids("pos"); neg_ids = movie_reviews.fileids("neg")
    def extract(words): return {word.lower(): True for word in words if word.isalpha()}
    pos = [(extract(movie_reviews.words(fileids=[f])), "Positive") for f in pos_ids]
    neg = [(extract(movie_reviews.words(fileids=[f])), "Negative") for f in neg_ids]
    rng = random.Random(42); rng.shuffle(pos); rng.shuffle(neg)
    np, nn = int(len(pos)*0.8), int(len(neg)*0.8)
    train = pos[:np] + neg[:nn]
    test = pos[np:] + neg[nn:]
    rng.shuffle(train); rng.shuffle(test)
    clf = NaiveBayesClassifier.train(train)
    acc = nltk_accuracy(clf, test)
    print(f"Обучающая выборка: {len(train)}; тестовая: {len(test)}; accuracy={acc:.4f}")
    print("20 наиболее информативных признаков:")
    import io, contextlib
    buff = io.StringIO()
    with contextlib.redirect_stdout(buff): clf.show_most_informative_features(20)
    informative = buff.getvalue().strip()
    print(informative)
    samples = [
        "A beautifully acted film with a moving story and excellent direction.",
        "The plot is dull, the dialogue is weak, and the ending is disappointing.",
        "An entertaining adventure with impressive visuals and memorable characters.",
        "The movie feels slow and predictable, despite a talented cast."
    ]
    sample_results = []
    for text in samples:
        dist = clf.prob_classify(extract(text.split()))
        item = {"text": text, "prediction": dist.max(),
                "probabilities": {label: round(dist.prob(label),4) for label in ["Positive","Negative"]}}
        sample_results.append(item)
        print(f"{item['prediction']} {item['probabilities']}: {text}")
    analysis = ("Качество оценивается на отдельной тестовой части корпуса movie_reviews. "
                "Наиболее информативные признаки показывают слова, сильнее всего связанные "
                "с классами. Для коротких вручную составленных предложений вероятности являются "
                "оценкой модели, а не гарантией правильности.")
    print("Анализ:", analysis)
    RESULTS["task8"] = {"train_size": len(train), "test_size": len(test),
                        "accuracy": acc, "informative_features": informative,
                        "samples": sample_results, "analysis": analysis}

def task9(nltk):
    section("ЗАДАНИЕ 9. ТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ LDA")
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    from gensim import models, corpora
    if not DATA_FILE.exists(): raise FileNotFoundError(f"Не найден {DATA_FILE}")
    raw = DATA_FILE.read_text(encoding="utf-8")
    tokenizer = RegexpTokenizer(r"[A-Za-z]+")
    stop = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    tokens = []
    for sentence in raw.splitlines():
        words = [stemmer.stem(w.lower()) for w in tokenizer.tokenize(sentence)]
        filtered = [w for w in words if w not in stop and len(w) > 2]
        if filtered: tokens.append(filtered)
    dictionary = corpora.Dictionary(tokens)
    doc_term_mat = [dictionary.doc2bow(token) for token in tokens]
    lda = models.ldamodel.LdaModel(corpus=doc_term_mat, num_topics=2,
                                   id2word=dictionary, passes=25, random_state=42,
                                   alpha="auto", eta="auto")
    topics = []
    for topic_id, words in lda.show_topics(num_topics=2, num_words=5, formatted=False):
        topic = [{"word": word, "weight": float(weight)} for word,weight in words]
        topics.append({"topic": int(topic_id), "words": topic})
        print(f"Тема {topic_id+1}: " + ", ".join(f"{w['word']} ({w['weight']:.3f})" for w in topic))
    analysis = ("LDA без учителя распределяет слова по двум скрытым темам. По исходным десяти "
                "предложениям ожидается разделение лексики историко-культурной тематики и "
                "математических понятий; конкретные веса зависят от инициализации и малого объёма корпуса.")
    print("Документов:", len(tokens), "| размер словаря:", len(dictionary))
    print("Анализ:", analysis)
    RESULTS["task9"] = {"documents": len(tokens), "vocabulary_size": len(dictionary),
                        "topics": topics, "analysis": analysis}

def task10_transformer():
    section("ЗАДАНИЕ 10. FINE-TUNING BERT-TINY (HUGGING FACE)")
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, TrainingArguments, Trainer)
    from sklearn.metrics import accuracy_score, f1_score

    model_name = "prajjwal1/bert-tiny"
    print("Задача: бинарная классификация тональности.")
    print("Датасет: GLUE / SST-2 (англоязычные предложения с метками Positive/Negative).")
    print("Модель:", model_name)

    dataset = load_dataset("glue", "sst2")
    pool = dataset["train"].shuffle(seed=42).select(range(min(6000, len(dataset["train"]))))
    split = pool.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    train_ds = split["train"]
    eval_ds = split["test"]
    # В GLUE/SST-2 метки официального test split скрыты; размеченный validation split
    # используется как независимая от обучения hold-out тестовая выборка.
    test_ds = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    keep = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep)
    eval_ds.set_format(type="torch", columns=keep)
    test_ds.set_format(type="torch", columns=keep)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def metrics(pred):
        logits, labels = pred
        pred_ids = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, pred_ids),
                "f1": f1_score(labels, pred_ids)}

    args = TrainingArguments(
        output_dir=str(ARTIFACTS / "bert_tiny_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=42,
        fp16=bool(torch.cuda.is_available()),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=eval_ds, tokenizer=tokenizer,
        data_collator=collator, compute_metrics=metrics
    )
    trainer.train()
    validation_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    all_metrics = {**validation_metrics, **test_metrics}

    probes = [
        "I absolutely loved this film; it was funny and moving.",
        "The movie was boring and I regret watching it.",
        "A thoughtful story with excellent performances.",
        "Terrible acting and a painfully predictable plot."
    ]
    inputs = tokenizer(probes, truncation=True, padding=True, max_length=128, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    labels = ["negative", "positive"]
    test_predictions = []
    for text, prob in zip(probes, probs):
        item = {
            "text": text,
            "prediction": labels[int(prob.argmax())],
            "probabilities": {"negative": float(prob[0]), "positive": float(prob[1])}
        }
        test_predictions.append(item)
        print(f"{item['prediction']} (neg={prob[0]:.3f}, pos={prob[1]:.3f}): {text}")

    model_dir = ARTIFACTS / "bert_tiny_sst2"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("Метрики validation:", validation_metrics)
    print("Метрики hold-out test:", test_metrics)
    print("Модель сохранена:", model_dir)

    RESULTS["task10"] = {
        "task": "Бинарная классификация тональности предложений",
        "dataset": "GLUE / SST-2",
        "dataset_description": (
            "Англоязычный набор предложений из отзывов о фильмах с бинарными метками: "
            "0 — Negative, 1 — Positive."
        ),
        "test_split_description": (
            "Для независимого тестирования использован официальный размеченный validation split SST-2 "
            "(872 примера), поскольку метки официального test split в GLUE не опубликованы."
        ),
        "model": model_name,
        "model_description": (
            "BERT-tiny — компактная предварительно обученная модель семейства BERT, "
            "используемая для классификации последовательности из двух классов."
        ),
        "training_procedure": (
            "AutoTokenizer; усечение последовательностей до 128 токенов; стратифицированное "
            "разбиение 6000 обучающих примеров на 80% train и 20% validation; "
            "Trainer/AdamW; learning rate 3e-5; batch size 8; weight decay 0.01; "
            "3 эпохи; seed=42."
        ),
        "train_examples": len(train_ds),
        "validation_examples": len(eval_ds),
        "test_examples": len(test_ds),
        "epochs": 3,
        "metrics": {
            k: float(v) for k, v in all_metrics.items()
            if isinstance(v, (int, float))
        },
        "predictions": test_predictions,
        "analysis": (
            "Validation-метрики измеряют качество на отложенной стратифицированной части обучающего "
            "набора; test-метрики — на отдельном размеченном hold-out split SST-2. "
            "Дополнительные предложения не входят в датасет и проверяют инференс на новых текстах. "
            "Вероятности показывают уверенность классификатора, но не являются гарантией правильности."
        )
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transformer", action="store_true",
                        help="не запускать загрузку/дообучение BERT-tiny")
    parser.add_argument("--only-transformer", action="store_true",
                        help="запустить только задание 10, сохранив результаты заданий 1–9")
    args = parser.parse_args()
    print("Лабораторная работа №2 — NLP | Python", sys.version.split()[0])
    print("Рабочая папка:", BASE)
    if args.only_transformer:
        previous = BASE / "lab02_results.json"
        if previous.exists():
            RESULTS.update(json.loads(previous.read_text(encoding="utf-8")))
        try:
            task10_transformer()
        except Exception as exc:
            RESULTS["task10"] = {"status": "ERROR", "error": repr(exc)}
            traceback.print_exc()
        finally:
            save_results()
            print("Результаты сохранены в lab02_results.json")
        return
    nltk = ensure_nltk()
    try:
        task1(nltk); task2(); task3(nltk)
        chunks = task4(nltk); task5(chunks); task6(); task7(nltk); task8(nltk); task9(nltk)
        if not args.skip_transformer:
            try:
                task10_transformer()
            except Exception as exc:
                RESULTS["task10"] = {"status": "ERROR", "error": repr(exc),
                                     "analysis": "Fine-tuning не завершён; требуется доступ к Hugging Face и совместимые версии библиотек."}
                print("\nОШИБКА ЗАДАНИЯ 10:", repr(exc))
                traceback.print_exc()
        else:
            RESULTS["task10"] = {"status": "SKIPPED_BY_ARGUMENT"}
    except Exception as exc:
        print("\nКРИТИЧЕСКАЯ ОШИБКА:", repr(exc))
        traceback.print_exc()
        RESULTS["fatal_error"] = repr(exc)
    finally:
        save_results()
        print("\nРезультаты сохранены в lab02_results.json")
        print("Готовые артефакты:", ARTIFACTS)

if __name__ == "__main__":
    main()