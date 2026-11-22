from pathlib import Path

here = Path(__file__).resolve().parent
parts = sorted(here.glob("model.safetensors.part*"))
if not parts:
    raise SystemExit("Не найдены части model.safetensors.part*")
target = here / "model.safetensors"
with target.open("wb") as output:
    for part in parts:
        output.write(part.read_bytes())
print(f"Модель восстановлена: {target} ({target.stat().st_size:,} байт)")
