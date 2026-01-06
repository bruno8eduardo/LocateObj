import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Carregar o CSV
arquivo_csv = r"tests\UFRJ\results\R_from_yaw_pitch_roll.csv"  # Substitua pelo nome do seu arquivo
df = pd.read_csv(arquivo_csv, sep=';')

# Garantir que a primeira coluna seja usada no eixo X
df[df.columns[0]] = df[df.columns[0]].astype(str)  # Converter para string caso seja número
df.set_index(df.columns[0], inplace=True)  # Definir a primeira coluna como índice

# Converter colunas numéricas (caso tenham sido lidas como string)
# for coluna in df.columns:
#     df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

# Detectar colunas no formato "(x, y)" e parsear
def parse_tuple_cell(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    nums = re.findall(r'-?\d+\.?\d*', str(s))
    if len(nums) >= 2:
        return (float(nums[0]), float(nums[1]))
    return (np.nan, np.nan)

tuple_cols = []
for coluna in df.columns:
    # verifica se existe pelo menos um valor no estilo "(num, num)"
    sample = df[coluna].dropna().astype(str)
    if sample.str.match(r'^\s*\(\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\)\s*$').any():
        tuple_cols.append(coluna)
        parsed = sample.apply(parse_tuple_cell)
        # criar colunas separadas x e y (mantém NaNs)
        full_parsed = df[coluna].astype(str).apply(parse_tuple_cell)
        df[f"{coluna}_x"] = full_parsed.apply(lambda t: t[1])
        df[f"{coluna}_y"] = full_parsed.apply(lambda t: t[0])

if tuple_cols:
    col = tuple_cols[0]  # se tiver mais de uma, pega a primeira; ajuste conforme necessário
    xcol = f"{col}_x"
    ycol = f"{col}_y"
    # Estimar centro se não informado
    cx = 1080 / 2.0
    cy = 1920 / 2.0
    df["Pixels"] = np.hypot(df[xcol] - cx, df[ycol] - cy)

# Criar duas figuras
fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)  # Dois gráficos na vertical

# Gráfico 1: Somente a Coluna 2
axs[0].plot(df.index, df.iloc[:, 0], label=df.columns[0], color="red")
axs[0].set_ylabel("m")
axs[0].set_title("Erro de Posicionamento (m)")
axs[0].legend()
axs[0].grid(True)

# Gráfico 1: Somente a Coluna 4
axs[2].plot(df.index, df.iloc[:, 2], label=df.columns[2], color="blue")
axs[2].set_ylabel("m")
axs[2].set_title("Distância do Drone (m)")
axs[2].legend()
axs[2].grid(True)

# Gráfico 1: Somente a Coluna 5
axs[4].plot(df.index, df.iloc[:, 3], label=df.columns[3], color="orange")
axs[4].set_ylabel("")
axs[4].set_title("Zoom")
axs[4].legend()
axs[4].grid(True)

# Gráfico 1: Somente a Coluna 6
axs[1].plot(df.index, df.iloc[:, 4], label=df.columns[4], color="brown")
axs[1].set_ylabel("m/s")
axs[1].set_title("Velocidade Estimada (m/s)")
axs[1].legend()
axs[1].grid(True)

if tuple_cols:
    radial_col = "Pixels"
    axs[3].plot(df.index, df[radial_col], label=radial_col, color="green")
    axs[3].set_ylabel("pixels")
    axs[3].set_title("Distância do Centro da Imagem (pixels)")
    axs[3].legend()
    axs[3].grid(True)
else:
    axs[3].plot(df.index, df.iloc[:, 1], label=df.columns[1], color="green")
    axs[3].set_ylabel("m")
    axs[3].set_title("Altura do Drone (m)")
    axs[3].legend()
    axs[3].grid(True)

# Melhorar visualização dos rótulos do eixo X
plt.xticks(rotation=45)

# Ajustar espaçamento entre gráficos
plt.tight_layout()

# Mostrar os gráficos
plt.show()