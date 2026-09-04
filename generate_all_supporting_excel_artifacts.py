from pathlib import Path
import shutil
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

TARGETS = [
    ROOT / "evaluation" / "data" / "operational",
    ROOT / "evaluation" / "data" / "risk_economic",
    ROOT / "evaluation" / "data" / "vpp_performance",
    ROOT / "evaluation" / "data" / "case_studies",
    ROOT / "evaluation" / "data" / "training_diagnostics",
    ROOT / "evaluation" / "data" / "forecast_horizon",
    ROOT / "forecasting" / "confidence_data",
    ROOT / "forecasting" / "uncertainty_data",
    ROOT / "data" / "publication",
]

FINAL_FILES = {
    "operational": "fc_hmarl_operational_dynamics.xlsx",
    "risk_economic": "risk_revenue_sensitivity_data.xlsx",
    "vpp_performance": "vpp_economic_operational_performance.xlsx",
    "case_studies": "vpp_case_study_comparison_data.xlsx",
    "training_diagnostics": "fc_hmarl_training_convergence_data.xlsx",
    "forecast_horizon": "forecast_horizon_metrics.xlsx",
    "confidence_data": "forecast_confidence_evolution_data.xlsx",
}

def safe_numeric_summary(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame({"Message": ["No numeric columns available"]})
    return numeric.describe().T.reset_index().rename(columns={"index": "Variable"})

def validation_table(df):
    rows = []
    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        rows.append({
            "Column": col,
            "Rows": len(series),
            "Missing": int(series.isna().sum()),
            "Unique": int(series.nunique(dropna=True)),
            "Numeric_Values": int(numeric.notna().sum()),
            "Minimum": float(numeric.min()) if numeric.notna().any() else np.nan,
            "Maximum": float(numeric.max()) if numeric.notna().any() else np.nan,
            "Mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
            "Std": float(numeric.std()) if numeric.notna().sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)

def find_source(folder):
    preferred = FINAL_FILES.get(folder.name)
    if preferred:
        candidate = folder / preferred
        if candidate.exists():
            return candidate

    files = sorted(
        p for p in folder.glob("*.xlsx")
        if not p.name.startswith(("01_", "02_", "03_", "04_"))
        and not p.name.startswith("~$")
    )
    return files[0] if files else None

def write_workbook(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

def process_folder(folder):
    if not folder.exists():
        return f"SKIP  {folder.relative_to(ROOT)}  folder not found"

    source = find_source(folder)
    if source is None:
        return f"SKIP  {folder.relative_to(ROOT)}  no source .xlsx found"

    prefix = folder.name

    raw_path = folder / f"01_{prefix}_source_snapshot.xlsx"
    cleaned_path = folder / f"02_{prefix}_cleaned_dataset.xlsx"
    stats_path = folder / f"03_{prefix}_summary_statistics.xlsx"
    validation_path = folder / f"04_{prefix}_validation_report.xlsx"

    shutil.copy2(source, raw_path)

    xls = pd.ExcelFile(source)
    cleaned_sheets = {}
    stats_sheets = {}
    validation_sheets = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(source, sheet_name=sheet)
        cleaned = df.copy()

        for col in cleaned.columns:
            if cleaned[col].dtype == object:
                cleaned[col] = cleaned[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

        cleaned_sheets[sheet] = cleaned
        stats_sheets[sheet] = safe_numeric_summary(cleaned)
        validation_sheets[sheet] = validation_table(cleaned)

    write_workbook(cleaned_path, cleaned_sheets)
    write_workbook(stats_path, stats_sheets)
    write_workbook(validation_path, validation_sheets)

    return (
        f"OK    {folder.relative_to(ROOT)}\n"
        f"      source: {source.name}\n"
        f"      + {raw_path.name}\n"
        f"      + {cleaned_path.name}\n"
        f"      + {stats_path.name}\n"
        f"      + {validation_path.name}"
    )

print("=" * 78)
print("FC-HMARL SUPPORTING EXCEL ARTIFACT GENERATOR")
print("=" * 78)

for target in TARGETS:
    try:
        print(process_folder(target))
    except Exception as exc:
        print(f"ERROR {target.relative_to(ROOT)}: {exc}")

print("=" * 78)
print("Finished.")
print("The original final Excel workbooks were not modified.")
print("=" * 78)
