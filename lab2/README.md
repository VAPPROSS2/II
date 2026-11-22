# Лабораторная работа №2 — NLP

Полный проект лабораторной работы №2. Работа выполнена в Visual Studio Code на Python 3.11.

## Состав

- `lab02.py` — задания 1–10;
- `lab02_data.txt` — исходный текст для тематического моделирования;
- `lab02_results.json` — результаты выполнения;
- `requirements.txt` — зависимости;
- `make_report.py` — генератор отчёта DOCX;
- `show_results.py` — просмотр результатов в терминале;
- `Отчет_ЛР2_NLP.docx` — отчёт;
- `screenshots/` — скриншоты выполнения в VS Code;
- `artifacts/bert_tiny_sst2/` — конфигурация, токенизатор и сохранённая модель BERT-tiny.

## Запуск

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -u lab02.py
py -3.11 make_report.py
```

Для повторного запуска только задания 10 после выполнения заданий 1–9:

```powershell
py -3.11 -u lab02.py --only-transformer
```

Результаты сохраняются в `lab02_results.json`. Для просмотра:
```powershell
py -3.11 show_results.py --section 1
py -3.11 show_results.py --section 2
py -3.11 show_results.py --section 3
```

Модель: `prajjwal1/bert-tiny`; датасет: GLUE/SST-2. Используются 6000 обучающих примеров, 3 эпохи, стратифицированное разбиение train/validation и отдельная оценка на размеченном официальном validation split SST-2 (поскольку метки официального test split не опубликованы).
