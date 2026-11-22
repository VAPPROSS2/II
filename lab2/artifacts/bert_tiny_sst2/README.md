# BERT-tiny model files

Файл весов `model.safetensors` разбит на последовательные бинарные части `model.safetensors.part00` … `model.safetensors.part11`, чтобы загрузить его в репозиторий через GitHub API без потери байтов.

После клонирования репозитория из корня проекта выполните:

```powershell
py -3.11 lab2/artifacts/bert_tiny_sst2/rebuild_model.py
```

Скрипт объединит части в исходный `model.safetensors`. После этого каталог можно использовать как сохранённую модель Hugging Face.
