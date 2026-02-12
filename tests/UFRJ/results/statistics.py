import pandas as pd

# Carregar o CSV usando ";" como separador
arquivo_csv = r"tests\UFRJ\results\R_from_yaw_pitch_roll.csv"  # Substitua pelo nome do seu arquivo
df = pd.read_csv(arquivo_csv, sep=";")

# Converter a primeira coluna para numérico (caso seja string) e definir como índice
df[df.columns[0]] = pd.to_numeric(df[df.columns[0]], errors="coerce")
df.set_index(df.columns[0], inplace=True)  # Definir como índice

# Converter as colunas 2 e 4 para numérico
coluna2 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
coluna4 = pd.to_numeric(df.iloc[:, 2], errors="coerce")  # Coluna 4 (índice 2 pois começa em 0)

# Selecionar os intervalos para a Erro
intervalo1_col2 = coluna2.loc[1:1651]  # Índices de 1 a 1651
intervalo2_col2 = coluna2.loc[2051:3801]  # Índices de 2051 a 3801
intervalo3_col2 = coluna2.loc[4351:4751]  # Índices de 4351 a 4751

# Selecionar os intervalos para a Coluna 4
intervalo1_col4 = coluna4.loc[1:1651]  # Índices de 1 a 1651
intervalo2_col4 = coluna4.loc[2051:3801]  # Índices de 2051 a 3801
intervalo3_col4 = coluna4.loc[4351:4751]  # Índices de 4351 a 4751

# Calcular média e máximo para a Erro
media1 = intervalo1_col2.mean()
maximo1 = intervalo1_col2.max()
desv_padr1 = intervalo1_col2.std()

media2 = intervalo2_col2.mean()
maximo2 = intervalo2_col2.max()
desv_padr2 = intervalo2_col2.std()

media3 = intervalo3_col2.mean()
maximo3 = intervalo3_col2.max()
desv_padr3 = intervalo3_col2.std()

# Calcular as medias da Coluna 4
media1_col4 = intervalo1_col4.mean()
media2_col4 = intervalo2_col4.mean()
media3_col4 = intervalo3_col4.mean()

# Exibir os resultados
print(f"Intervalo 1 (1 a 1651): Distância Média = {media1_col4:.4f}, Média Erro = {media1:.4f}, Máximo Erro = {maximo1}, Desvio Padrao Erro: {desv_padr1}")
print(f"Intervalo 2 (2051 a 3801): Distância Média = {media2_col4:.4f}, Média Erro = {media2:.4f}, Máximo Erro = {maximo2}, Desvio Padrao Erro: {desv_padr2}")
print(f"Intervalo 3 (4351 a 4751): Distância Média = {media3_col4:.4f}, Média Erro = {media3:.4f}, Máximo Erro = {maximo3}, Desvio Padrao Erro: {desv_padr3}")

