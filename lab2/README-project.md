# Лабораторная работа №2 — NLP

Проект выполнен в Visual Studio Code на Python 3.11.9.

## Состав проекта

- `lab02.py` — реализация заданий 1–10;
- `lab02_data.txt` — исходный файл для тематического моделирования;
- `lab02_results.json` — сохранённые фактические результаты запуска;
- `make_report.py` — генерация отчёта DOCX с таблицами и скриншотами;
- `show_results.py` — вывод сохранённых результатов в терминал VS Code;
- `artifacts/bert_tiny_sst2/` — дообученная модель BERT-tiny и токенизатор;
- `screenshots/` — реальные скриншоты выполнения в VS Code.

## Запуск

В терминале VS Code из каталога проекта:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -u lab02.py
py -3.11 make_report.py
```

Полный запуск выполняет задания 1–10. Для задания 10 загружается GLUE/SST-2, 6000 обучающих примеров делятся стратифицированно на 80% train и 20% validation; официальный размеченный validation split SST-2 (872 примера) используется как независимая hold-out тестовая выборка, поскольку метки официального test split не опубликованы. Модель `prajjwal1/bert-tiny` дообучается 3 эпохи; рассчитываются Accuracy и F1 для validation и test. Модель сохраняется в `artifacts/bert_tiny_sst2/`.

Если задания 1–9 уже выполнены и требуется повторно запустить только fine-tuning:

```powershell
py -3.11 -u lab02.py --only-transformer
```

Результаты всех заданий сохраняются в `lab02_results.json`. Для просмотра:

```powershell
py -3.11 show_results.py --section 1
py -3.11 show_results.py --section 2
py -3.11 show_results.py --section 3
```

Отчёт: `Отчет_ЛР2_NLP.docx`.

[executed on device: HOME-PC (e6e0ac5b-fa1d-4b2a-ae2e-d82fd4bf130e)]