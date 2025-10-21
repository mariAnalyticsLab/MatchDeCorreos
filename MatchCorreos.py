from rapidfuzz import process, fuzz
import unidecode, re
import numpy as np
import pandas as pd
import time
import os
import csv
inicio = time.time()
#Abrir y crear df de cada correo y autor
df = pd.read_csv(r"C:\Users\maria\OneDrive\Documents\UNIIE\NoEncontrados.csv",header=0)
# dfs=pd.read_excel(r"C:\Users\maria\OneDrive\Documents\UNIIE\autores.xlsx",header=0)
# dfa=pd.read_csv(r"C:\Users\maria\OneDrive\Documents\UNIIE\nombres_orcid_completo.csv",header=0)
dfs=pd.read_csv(r"C:\Users\maria\OneDrive\Documents\UNIIE\MasNombresInvestigadores.csv",header=0)
columna=list(df['CorreoParecido'])
fila=list(dfs['Nombre']) 
# otros=list(dfa['Nombre'])
# fila=filas+otros #Combinación de los nombres de autores
df_match = pd.DataFrame(columns=['Nombre', 'Correo',"Similitud"])

inicio = time.time()

# --- FUNCIONES ---
def normalizar(t):
    return re.sub(r'[^a-z]', '', unidecode.unidecode(t.lower().strip()))

def limpiar_correo(c):
    return normalizar(re.sub(r'\d+', '', c.split('@')[0]))

# --- DATOS ---
# fila: lista de nombres
# columna: lista de correos
nombres_norm = [normalizar(n) for n in fila]
correos_limpios = [limpiar_correo(c) for c in columna]
correos_reales = list(columna)

# --- CÁLCULO DE SIMILITUD EN BLOQUE ---
scores = process.cdist(
    nombres_norm,
    correos_limpios,
    scorer=fuzz.token_set_ratio
)

best_idx = np.argmax(scores, axis=1)
best_scores = np.max(scores, axis=1)

UMBRAL = 90

# --- RESULTADOS PRINCIPALES (similitud >= UMBRAL) ---
resultados = [
    (fila[i], correos_reales[best_idx[i]], best_scores[i])
    for i in range(len(fila))
    if best_scores[i] >= UMBRAL
]

df_match = pd.DataFrame(resultados, columns=["Nombre", "Correo", "Similitud"])

# --- NO ENCONTRADOS (similitud < UMBRAL) ---
no_encontrados = [
    (fila[i], correos_reales[best_idx[i]], best_scores[i])
    for i in range(len(fila))
    if best_scores[i] < UMBRAL
]

df_no = pd.DataFrame(no_encontrados, columns=["Nombre", "Correo más parecido", "Similitud"])

# --- GUARDAR ARCHIVOS ---
os.chdir(r"C:/Users/maria/OneDrive/Documents/UNIIE")
df_match.to_csv("MatchCorreos.csv", mode="a", index=False, encoding='utf-8-sig')
df_no.to_csv("NoEncontrados.csv", index=False, encoding='utf-8-sig')

# --- INFORMACIÓN ---
print(f"Archivo 'MatchCorreos.csv' guardado con {len(df_match)} registros")
print(f"Archivo 'NoEncontrados.csv' guardado con {len(df_no)} registros")
print(f"\nDuración total: {time.time() - inicio:.2f} segundos")
