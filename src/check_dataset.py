from pathlib import Path
import os

base_path = Path("dataset")

if not base_path.exists():
    raise FileNotFoundError("dataset 폴더가 없습니다. Kaggle 데이터셋 압축을 dataset 폴더 안에 풀었는지 확인하세요.")

for root, dirs, files in os.walk(base_path):
    level = root.replace(str(base_path), "").count(os.sep)

    if level > 4:
        continue

    indent = "  " * level
    print(f"{indent}{Path(root).name}/")

    for f in files[:8]:
        print(f"{indent}  {f}")