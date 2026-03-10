import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_CSV = r"tests\UFRJ\results\R_from_yaw_pitch_roll.csv"
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920


def parse_tuple_cell(value):
    if pd.isna(value):
        return (np.nan, np.nan)

    numbers = re.findall(r"-?\d+\.?\d*", str(value))
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))

    return (np.nan, np.nan)


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()

    frame_column = df.columns[0]
    df[frame_column] = pd.to_numeric(df[frame_column], errors="coerce")
    df.set_index(frame_column, inplace=True)

    tuple_columns = []
    for column in df.columns:
        series = df[column].dropna().astype(str).str.strip()
        if series.str.match(r"^\(\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\)$").any():
            tuple_columns.append(column)
            parsed = df[column].apply(parse_tuple_cell)
            df[f"{column}_x"] = parsed.apply(lambda item: item[1])
            df[f"{column}_y"] = parsed.apply(lambda item: item[0])

    for column in df.columns:
        if column in tuple_columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if tuple_columns:
        pixel_column = tuple_columns[0]
        cx = FRAME_WIDTH / 2.0
        cy = FRAME_HEIGHT / 2.0
        df["Pixels"] = np.hypot(df[f"{pixel_column}_x"] - cx, df[f"{pixel_column}_y"] - cy)

    return df


def build_correlation_table(df):
    preferred_columns = [
        "Erro",
        "Altura do Drone",
        "Distância do Drone",
        "Zoom",
        "Velocidade",
        "Velocidade Angular",
        "Pixels",
    ]
    available_columns = [column for column in preferred_columns if column in df.columns]

    numeric_df = df[available_columns].dropna(how="all")
    pearson = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")
    pearson_pvalues = build_pvalue_matrix(numeric_df, method="pearson")
    spearman_pvalues = build_pvalue_matrix(numeric_df, method="spearman")

    return available_columns, pearson, spearman, pearson_pvalues, spearman_pvalues


def build_pvalue_matrix(df, method):
    columns = df.columns
    pvalues = pd.DataFrame(np.nan, index=columns, columns=columns)

    for row in columns:
        for column in columns:
            valid = df[[row, column]].dropna()
            if len(valid) < 2:
                continue

            if row == column:
                pvalues.loc[row, column] = 0.0
                continue

            if method == "pearson":
                _, pvalue = pearsonr(valid[row], valid[column])
            elif method == "spearman":
                _, pvalue = spearmanr(valid[row], valid[column])
            else:
                raise ValueError(f"Metodo de correlacao invalido: {method}")

            pvalues.loc[row, column] = pvalue

    return pvalues


def print_error_focus(correlation_matrix, pvalue_matrix, method_name, reference_column="Erro"):
    if reference_column not in correlation_matrix.columns:
        return

    print(f"\n{method_name}: correlacao de {reference_column} com as outras variaveis:")
    ordered = correlation_matrix[reference_column].drop(reference_column).sort_values(
        key=lambda values: values.abs(),
        ascending=False,
    )
    for column, value in ordered.items():
        pvalue = pvalue_matrix.loc[column, reference_column]
        print(f"  {reference_column} x {column}: correlacao = {value:.4f}, p-value = {pvalue:.4e}")


def format_matrix(matrix):
    return matrix.apply(lambda column: column.map(
        lambda value: f"{value:.4e}" if pd.notna(value) else "nan"
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Calcula a correlacao entre os dados do arquivo de resultados."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Caminho do CSV de entrada. Padrao: {DEFAULT_CSV}",
    )
    args = parser.parse_args()

    df = load_dataframe(args.csv)
    columns, pearson, spearman, pearson_pvalues, spearman_pvalues = build_correlation_table(df)

    if len(columns) < 2:
        raise ValueError("Nao ha colunas numericas suficientes para calcular correlacao.")

    print("Colunas usadas na correlacao:")
    for column in columns:
        print(f"  - {column}")

    print("\nMatriz de correlacao de Pearson:")
    print(pearson.round(4).to_string())

    print("\nMatriz de p-value de Pearson:")
    print(format_matrix(pearson_pvalues).to_string())

    print("\nMatriz de correlacao de Spearman:")
    print(spearman.round(4).to_string())

    print("\nMatriz de p-value de Spearman:")
    print(format_matrix(spearman_pvalues).to_string())

    print_error_focus(pearson, pearson_pvalues, "Pearson", reference_column="Erro")
    print_error_focus(spearman, spearman_pvalues, "Spearman", reference_column="Erro")



if __name__ == "__main__":
    main()
