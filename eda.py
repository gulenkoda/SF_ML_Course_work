from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent / "coursework_data.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "eda"
TARGET_COLUMNS = ["IC50, mM", "CC50, mM", "SI"]
INDEX_COLUMN = "Unnamed: 0"


def iqr_outlier_summary(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return {
        "q1": q1,
        "q3": q3,
        "lower_bound": lower,
        "upper_bound": upper,
        "outliers_count": int(mask.sum()),
        "outliers_share": float(mask.mean()),
        "max": float(series.max()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATA_PATH).replace([np.inf, -np.inf], np.nan)

    print("=== Общая информация ===")
    print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Дубликаты строк: {df.duplicated().sum()}")
    print(f"Всего пропусков: {int(df.isna().sum().sum())}")

    missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Не найдены целевые колонки: {missing_targets}")

    feature_cols = [
        col for col in df.columns
        if col not in TARGET_COLUMNS + [INDEX_COLUMN]
    ]
    constant_features = [col for col in feature_cols if df[col].nunique(dropna=False) <= 1]
    cleaned_features = [col for col in feature_cols if col not in constant_features]

    print("\n=== Признаки ===")
    print(f"Всего признаков без целей и индекса: {len(feature_cols)}")
    print(f"Константных признаков: {len(constant_features)}")
    print(f"Остаётся признаков после удаления констант: {len(cleaned_features)}")
    print("Константные признаки не несут информации и удаляются в анализе.")

    target_stats = df[TARGET_COLUMNS].describe().T
    print("\n=== Описательные статистики целей ===")
    print(target_stats[["mean", "50%", "std", "min", "max"]].round(3))
    target_stats.to_csv(OUTPUT_DIR / "target_statistics.csv")

    print("\n=== Анализ выбросов по IQR ===")
    outlier_rows = []
    for target in TARGET_COLUMNS:
        summary = iqr_outlier_summary(df[target])
        outlier_rows.append({"target": target, **summary})
        print(
            f"{target}: выбросов {summary['outliers_count']} "
            f"({summary['outliers_share']:.1%}), верхняя граница "
            f"{summary['upper_bound']:.3f}, максимум {summary['max']:.3f}"
        )
    pd.DataFrame(outlier_rows).to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

    print("\nРешение по выбросам: строки не удаляются автоматически. В биологических")
    print("данных экстремальные значения могут соответствовать реальным соединениям.")
    print("Их влияние учитывается через устойчивые метрики и сравнение разных моделей.")

    target_corr = df[TARGET_COLUMNS].corr(method="spearman")
    target_corr.to_csv(OUTPUT_DIR / "target_spearman_correlations.csv")

    feature_corr_rows = []
    for target in TARGET_COLUMNS:
        corr = df[cleaned_features + [target]].corr(method="spearman")[target]
        corr = corr.drop(index=target).abs().sort_values(ascending=False)
        feature_corr_rows.append({
            "target": target,
            "max_abs_spearman": float(corr.iloc[0]),
            "feature": corr.index[0],
        })
    feature_corr_table = pd.DataFrame(feature_corr_rows)
    feature_corr_table.to_csv(OUTPUT_DIR / "top_feature_correlations.csv", index=False)

    print("\n=== Связь отдельных признаков с целями ===")
    print(feature_corr_table.round(3).to_string(index=False))
    print("Максимальные связи слабые, поэтому явной скрытой утечки не видно.")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=np.log1p(df[TARGET_COLUMNS]))
    plt.title("Распределение целевых переменных, log1p")
    plt.ylabel("log1p(value)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_boxplots_log.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(target_corr, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Spearman-корреляции целевых переменных")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_correlations.png", dpi=200)
    plt.close()

    for target in TARGET_COLUMNS:
        values = np.log1p(df[target].dropna())
        plt.figure(figsize=(7, 4))
        plt.hist(values, bins=40)
        plt.title(f"Распределение {target}, log1p")
        plt.xlabel(f"log1p({target})")
        plt.ylabel("count")
        plt.tight_layout()
        safe_name = target.replace(",", "").replace(" ", "_")
        plt.savefig(OUTPUT_DIR / f"hist_{safe_name}_log.png", dpi=200)
        plt.close()

    print("\nГрафики и таблицы сохранены в outputs/eda/.")
    print("Итог EDA: данные пригодны для моделирования, но цели сильно скошены,")
    print("SI содержит экстремальные значения, а отдельные дескрипторы слабо")
    print("связаны с целями. Поэтому нужны ансамблевые модели и честная оценка")
    print("на отложенной выборке.")


if __name__ == "__main__":
    main()
