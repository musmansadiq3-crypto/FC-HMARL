from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_FOLDER = PROJECT_ROOT / "configs"


CONFIG_FILES = [
    "system_config.yaml",
    "forecasting_config.yaml",
    "marl_config.yaml",
    "market_config.yaml",
]


print("=" * 70)
print("FC-HMARL CONFIGURATION CHECK")
print("=" * 70)


for filename in CONFIG_FILES:

    file_path = CONFIG_FOLDER / filename

    print(f"\nChecking: {file_path}")

    if not file_path.exists():

        print("ERROR: File does not exist.")

        continue

    try:

        with open(file_path, "r", encoding="utf-8") as file:

            data = yaml.safe_load(file)

        print("STATUS: OK")

        print("Top-level sections:")

        if isinstance(data, dict):

            for key in data.keys():

                print(f"   - {key}")

    except Exception as error:

        print("STATUS: ERROR")

        print(error)


print("\n" + "=" * 70)
print("CONFIGURATION CHECK COMPLETE")
print("=" * 70)