# -*- coding: utf-8 -*-
"""Формирование отчёта ЛР2 из фактических результатов выполнения lab02.py."""
from pathlib import Path
import json
from docx import Document
from docx.shared import Cm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = Path(__file__).resolve().parent
data = json.loads((BASE / "lab02_results.json").read_text(encoding="utf-8"))
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin, sec.bottom_margin = Cm(2), Cm(2)
sec.left_margin, sec.right_margin = Cm(3), Cm(1.5)

# Основной стиль отчёта
for name in ["Normal", "Title", "Heading 1", "Heading 2"]:
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
doc.styles["Normal"].font.size = Pt(14)
doc.styles["Normal"].paragraph_format.line_spacing = 1.5
doc.styles["Normal"].paragraph_format.first_line_indent = Cm(1.25)
for name in ["Heading 1", "Heading 2"]:
    st = doc.styles[name]
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.name = "Times New Roman"
    st.font.size = Pt(14)

# Номер страницы в нижнем колонтитуле
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), "PAGE")
footer._p.append(fld)

def centered(text, size=12, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    return p

def paragraph(text="", bold=False, size=14, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(1.25) if indent else Cm(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    return p

def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.first_line_indent = Cm(0)
    for r in p.runs:
        r.font.name = "Times New Roman"
        r.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_table(headers, rows, font_size=14):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, value in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = str(value)
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in c.paragraphs:
            p.paragraph_format.first_line_indent = Cm(0)
            for r in p.runs:
                r.bold = True
                r.font.name = "Times New Roman"
                r.font.size = Pt(font_size)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[i].paragraphs:
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.line_spacing = 1.0
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(font_size)
    return t

# Титульный лист: оформление ИТМО, как в предыдущих лабораторных
centered("МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ", 11, True)
centered("НАЦИОНАЛЬНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ УНИВЕРСИТЕТ ИТМО", 12, True)
centered("ФАКУЛЬТЕТ ПРИКЛАДНОЙ ИНФОРМАТИКИ", 12, True)
for _ in range(5):
    doc.add_paragraph()
centered("ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ № 2", 16, True)
centered("РЕШЕНИЕ ЗАДАЧ NLP", 16, True)
centered("с помощью библиотек NLTK, Scikit-learn, Gensim и Hugging Face", 13)
for _ in range(4):
    doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
p.paragraph_format.first_line_indent = Cm(0)
p.add_run("Выполнил: студент группы К3320\n").bold = True
p.add_run("Аннин Макар Дмитриевич\n")
p.add_run("ИСУ 465018\n")
p.add_run("___________ / Аннин М. Д. / ___________\n")
p.add_run("Дата: «___» __________ 2026 г.\n\n")
p.add_run("Руководитель:\n").bold = True
p.add_run("Пикуль А. С.\n")
p.add_run("___________ / Пикуль А. С. / ___________\n")
p.add_run("Дата: «___» __________ 2026 г.")
for _ in range(4):
    doc.add_paragraph()
centered("Санкт-Петербург", 12)
centered("2026", 12)
doc.add_page_break()

heading("Цель и постановка работы", 1)
paragraph("Цель работы — получить практические навыки работы с современными архитектурами и инструментами обработки текстовой информации (NLP). В работе выполнены десять заданий: токенизация, стемминг, лемматизация, чанкинг, Bag of Words, классификация текстов, определение грамматического рода имён, сентимент-анализ, тематическое моделирование LDA и дообучение модели семейства BERT.")
heading("Программная среда", 1)
paragraph("Работа выполнена в Visual Studio Code на Python 3.11. Использованы NLTK, Scikit-learn, Gensim, PyTorch, Transformers и Datasets. Для задания 9 использован предоставленный файл lab02_data.txt.")

# 1
r = data.get("task1", {})
heading("1. Токенизация текстовых данных", 1)
paragraph("Исходный текст: " + r.get("text", ""))
for title, key in [("Токенизация предложений", "sentences"), ("Токенизация слов", "words"), ("WordPunctTokenizer", "wordpunct")]:
    heading(title, 2)
    paragraph(", ".join(r.get(key, [])), size=14)
paragraph("Анализ: " + r.get("analysis", ""))

# 2
r = data.get("task2", {})
heading("2. Стемминг", 1)
stemmers = r.get("stemmers", {})
add_table(["Исходное слово"] + list(stemmers.keys()),
          [[w] + [stemmers[k][i] for k in stemmers] for i, w in enumerate(r.get("words", []))], 14)
paragraph("Анализ: " + r.get("analysis", ""))

# 3
r = data.get("task3", {})
heading("3. Лемматизация", 1)
add_table(["Слово", "Существительное (n)", "Глагол (v)"],
          [[w, r.get("nouns", [])[i], r.get("verbs", [])[i]]
           for i, w in enumerate(r.get("words", []))], 14)
paragraph("Анализ: " + r.get("analysis", ""))

# 4
r = data.get("task4", {})
heading("4. Разбиение текста на чанки", 1)
paragraph(f"Из коллекции Brown взято {r.get('source_words')} слов. Размер чанка M={r.get('chunk_size')} слов; сформировано {r.get('chunk_count')} чанков. Ниже показаны первые 50 символов каждого чанка:")
for i, chunk in enumerate(r.get("chunks", []), 1):
    paragraph(f"Чанк {i}: {chunk[:50]}", size=14, indent=False)
paragraph("Анализ: " + r.get("analysis", ""))

# 5
r = data.get("task5", {})
heading("5. Извлечение частотности слов с помощью Bag of Words", 1)
paragraph(f"Размер терм-документной матрицы: {r.get('shape')}. Строки соответствуют чанкам, столбцы — словам словаря.")
terms, matrix, docs = r.get("terms", []), r.get("matrix", []), r.get("documents", [])
for start in range(0, len(terms), 5):
    group = terms[start:start+5]
    rows = [[docs[i]] + row[start:start+5] for i, row in enumerate(matrix)]
    add_table(["Чанк"] + group, rows, 14)
paragraph("Анализ: " + r.get("analysis", ""))

# 6
r = data.get("task6", {})
heading("6. Создание прогнозатора категорий", 1)
paragraph("Категории классификации: " + ", ".join(r.get("categories", [])))
paragraph(f"Обучающая выборка содержит {r.get('train_size')} документов. Использованы CountVectorizer, TF-IDF и MultinomialNB.")
add_table(["Тестовое предложение", "Предсказанная категория"],
          [[x["text"], x["predicted_category"]] for x in r.get("predictions", [])], 14)
paragraph("Анализ: " + r.get("analysis", ""))

# 7
r = data.get("task7", {})
heading("7. Создание анализатора грамматических родов", 1)
add_table(["Количество последних букв N", "Accuracy"],
          [[n, f"{v:.4f}"] for n, v in r.get("accuracy_by_n", {}).items()], 14)
for n, items in r.get("predictions", {}).items():
    paragraph(f"Предсказания при N={n}: " + ", ".join(f"{x['name']} — {x['prediction']}" for x in items), size=14)
paragraph("Анализ: " + r.get("analysis", ""))

# 8
r = data.get("task8", {})
heading("8. Создание сентимент-анализатора", 1)
paragraph(f"Обучающая выборка: {r.get('train_size')}; тестовая выборка: {r.get('test_size')}; accuracy: {r.get('accuracy', 0):.4f}.")
paragraph("20 наиболее информативных признаков:", bold=True)
for line in r.get("informative_features", "").splitlines()[1:]:
    paragraph(line, size=14, indent=False)
add_table(["Тестовый отзыв", "Тональность", "Вероятности"],
          [[x["text"], x["prediction"], str(x["probabilities"])] for x in r.get("samples", [])], 14)
paragraph("Анализ: " + r.get("analysis", ""))

# 9
r = data.get("task9", {})
heading("9. Тематическое моделирование с использованием LDA", 1)
paragraph(f"Количество документов: {r.get('documents')}; размер словаря: {r.get('vocabulary_size')}.")
for topic in r.get("topics", []):
    paragraph(f"Тема {topic['topic']+1}: " + ", ".join(f"{x['word']} ({x['weight']:.3f})" for x in topic["words"]))
paragraph("Анализ: " + r.get("analysis", ""))

# 10
r = data.get("task10", {})
heading("10. Самостоятельное решение задачи NLP с помощью Hugging Face", 1)
paragraph("Задача: " + r.get("task", ""))
paragraph("Датасет: " + r.get("dataset", ""))
paragraph("Описание датасета: " + r.get("dataset_description", ""))
paragraph("Описание модели: " + r.get("model_description", ""))
paragraph("Модель: " + r.get("model", ""))
paragraph("Процедура дообучения: " + r.get("training_procedure", ""))
paragraph(
    f"Обучающая выборка: {r.get('train_examples')}; валидационная: "
    f"{r.get('validation_examples')}; тестовая: {r.get('test_examples')}; "
    f"число эпох: {r.get('epochs')}."
)
paragraph("Метрики валидации:", bold=True)
metrics = r.get("metrics", {})
add_table(
    ["Метрика", "Значение"],
    [[k.replace("validation_", "").upper(), f"{v:.4f}"]
     for k, v in metrics.items()
     if k in ["validation_accuracy", "validation_f1", "validation_loss"]],
    14
)
paragraph("Метрики независимой тестовой выборки:", bold=True)
add_table(
    ["Метрика", "Значение"],
    [[k.replace("test_", "").upper(), f"{v:.4f}"]
     for k, v in metrics.items()
     if k in ["test_accuracy", "test_f1", "test_loss"]],
    14
)
paragraph("Примечание: в GLUE/SST-2 метки официального test split не опубликованы; поэтому в роли независимой тестовой выборки использован размеченный официальный validation split датасета. " + r.get("test_split_description", ""))
heading("Инференс на произвольных предложениях", 2)
add_table(
    ["Тестовое предложение", "Предсказание", "Вероятности"],
    [[x["text"], x["prediction"],
      str({k: round(v, 3) for k, v in x["probabilities"].items()})]
     for x in r.get("predictions", [])],
    14
)
paragraph("Анализ: " + r.get("analysis", ""))

# Скриншоты только из реального выполнения программы в VS Code
screens = sorted((BASE / "screenshots").glob("*.png")) if (BASE / "screenshots").exists() else []
if screens:
    heading("11. Скриншоты выполнения программы", 1)
    for img in screens:
        paragraph(img.stem.replace("_", " ").capitalize(), bold=True, indent=False)
        doc.add_picture(str(img), width=Inches(6.0))
        cap = doc.add_paragraph(f"Рисунок — {img.name}")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.first_line_indent = Cm(0)

heading("Вывод", 1)
paragraph("В ходе лабораторной работы реализованы базовые методы обработки естественного языка: токенизация, нормализация слов, разбиение документов, векторизация, классификация, определение тональности и тематическое моделирование. Эксперименты показали, что качество результатов зависит от выбора признаков, объёма обучающей выборки и настроек модели. Статистические методы позволяют построить интерпретируемые базовые решения, а дообученная модель семейства BERT учитывает контекст текста и классифицирует тональность новых предложений.")
heading("Исходный код", 1)
paragraph("Исходный код, данные и материалы лабораторной работы размещены в репозитории: https://github.com/VAPPROSS2/II/tree/main/lab02.")

out = BASE / "Отчет_ЛР2_NLP.docx"
doc.save(out)
print("DOCX:", out)