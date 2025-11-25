"""
script.py

Script en Python para analizar `Data_Biocompatibilidad.csv`.

Requisitos: pandas, matplotlib, matplotlib_venn, numpy

Instalación (si hace falta):
	pip install pandas matplotlib matplotlib-venn numpy

Ejecutar:
	python script.py

El script hace lo siguiente:
 - Carga el CSV con pandas
 - Muestra estadísticas básicas: materiales, medicamentos, reacciones, compatibilidad
 - Genera 3 gráficas: diagrama de Venn, barras top5 materiales, índice de compatibilidad

Comentarios están en español y pensados para nivel principiante.
"""

import os
import sys
import collections

# Intentamos importar las librerías necesarias y damos instrucciones si faltan
try:
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib_venn import venn2
except Exception as e:
	print("Falta alguna librería necesaria:", e)
	print("Instala dependencias con:")
	print("    pip install pandas matplotlib matplotlib-venn numpy")
	sys.exit(1)


def cargar_datos(ruta_csv):
	"""Carga el archivo CSV y devuelve un DataFrame de pandas.

	Maneja líneas donde la columna final `comentario` contiene comas sin comillas
	juntando cualquier columna extra en la última columna.
	"""
	import csv

	with open(ruta_csv, encoding='utf-8') as f:
		reader = csv.reader(f)
		rows = list(reader)

	if len(rows) == 0:
		return pd.DataFrame()

	header = rows[0]
	ncols = len(header)
	data = []

	# Recorremos cada fila y si tiene más columnas de las esperadas
	# unimos las columnas extras en la última columna (comentario).
	for i, row in enumerate(rows[1:], start=2):
		if len(row) < ncols:
			# rellenar con cadenas vacías si faltan columnas
			row = row + [''] * (ncols - len(row))
		elif len(row) > ncols:
			# unir las columnas que sobran en la última columna
			row = row[: ncols - 1] + [','.join(row[ncols - 1 :])]
		data.append(row)

	df = pd.DataFrame(data, columns=header)
	return df


def mostrar_estadisticas(df):
	"""Muestra estadísticas básicas por consola.

	- materiales más frecuentes
	- medicamentos más frecuentes
	- reacciones más frecuentes
	- conteo de compatibilidad
	"""
	print("\n=== Estadísticas básicas ===\n")

	# Materiales más frecuentes
	materiales = df['material'].dropna().astype(str)
	top_materiales = materiales.value_counts()
	print("Materiales más frecuentes (top 10):")
	print(top_materiales.head(10))

	# Medicamentos más frecuentes
	medicamentos = df['medicamento'].fillna('Ninguno').astype(str)
	top_medicamentos = medicamentos.value_counts()
	print("\nMedicamentos más frecuentes (top 10):")
	print(top_medicamentos.head(10))

	# Reacciones más frecuentes
	reacciones = df['reaccion'].fillna('Desconocida').astype(str)
	top_reacciones = reacciones.value_counts()
	print("\nReacciones más frecuentes (top 10):")
	print(top_reacciones.head(10))

	# Compatibilidad (conteo por categoría)
	if 'biocompatible' in df.columns:
		compat = df['biocompatible'].fillna('Desconocido')
		print("\nCompatibilidad (por categoría):")
		print(compat.value_counts())
	else:
		print("\nColumna 'biocompatible' no encontrada en el CSV.")


def diagrama_venn_latex_antibiotico(df):
	"""Crea un diagrama de Venn para:

	A = pacientes expuestos a material con 'látex'
	B = pacientes que recibieron un antibiótico (clasificacion_medicamento contiene 'Antibiótico')
	"""
	# Normalizamos texto para búsquedas sencillas
	material_lower = df['material'].fillna('').str.lower()

	# Detectamos exposición a látex buscando la palabra 'látex' o 'latex'
	exp_latex = material_lower.str.contains('l[aá]tex', regex=True)

	# Detectamos antibióticos por la columna 'clasificacion_medicamento'
	if 'clasificacion_medicamento' in df.columns:
		clasif_lower = df['clasificacion_medicamento'].fillna('').str.lower()
		recibio_antibiotico = clasif_lower.str.contains('antibiótico|antibiotico', regex=True)
	else:
		# Si no existe la columna, intentamos deducir por 'medicamento'
		med_lower = df['medicamento'].fillna('').str.lower()
		recibio_antibiotico = med_lower.str.contains('amoxicilina|penicilina|clindamicina|metronidazol', regex=True)

	# Conjuntos como IDs de paciente para evitar duplicados
	ids = df['id_paciente'] if 'id_paciente' in df.columns else df.index
	set_A = set(ids[exp_latex])
	set_B = set(ids[recibio_antibiotico])

	# Mostrar conteos por consola
	print("\n=== Diagrama de Venn (látex vs antibiótico) ===")
	print(f"Pacientes expuestos a látex (A): {len(set_A)}")
	print(f"Pacientes que recibieron antibiótico (B): {len(set_B)}")
	print(f"Intersección A ∩ B: {len(set_A & set_B)}")

	# Dibujar diagrama de Venn
	plt.figure(figsize=(6,6))
	venn = venn2([set_A, set_B], set_labels=('Expuestos a látex', 'Recibieron antibiótico'))
	plt.title('Diagrama de Venn: Látex vs Antibiótico')

	# Guardar la figura en la carpeta raíz del proyecto
	base_dir = os.path.dirname(__file__)
	path_venn = os.path.join(base_dir, 'venn_latex_antibiotico.png')
	plt.savefig(path_venn, dpi=300, bbox_inches='tight')
	print(f"Diagrama de Venn guardado en: {path_venn}")


def grafica_barras_top5_materiales(df):
	"""Grafica barras con los 5 materiales más utilizados."""
	materiales = df['material'].dropna().astype(str)
	top5 = materiales.value_counts().head(5)

	print("\n=== Top 5 materiales más utilizados ===")
	print(top5)

	plt.figure(figsize=(8,5))
	ax = top5.plot(kind='bar', color='skyblue')
	ax.set_title('Top 5 materiales más utilizados')
	ax.set_xlabel('Material')
	ax.set_ylabel('Conteo')
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()

	# Guardar la figura en la carpeta raíz del proyecto
	base_dir = os.path.dirname(__file__)
	path_top5 = os.path.join(base_dir, 'top5_materiales.png')
	plt.savefig(path_top5, dpi=300, bbox_inches='tight')
	print(f"Gráfica Top5 materiales guardada en: {path_top5}")


def indice_compatibilidad_por_medicamento(df, top_n=5):
	"""Calcula y grafica el índice de compatibilidad usando teoría de conjuntos.

	Definimos A = pacientes con antecedente de alergia (Sí) o gravedad Moderada/Grave.
	Para cada uno de los top N medicamentos más frecuentes (columna 'medicamento') se define B.
	Compatibilidad = |A ∩ B| / |A|
	"""
	# Construimos A
	antecedente = df['antecedente_alergia'].fillna('').astype(str).str.lower()
	tiene_antecedente = antecedente.str.contains('s', regex=False)  # coincide con 'Sí'

	gravedad = df['gravedad'].fillna('').astype(str).str.lower()
	es_moderada_o_grave = gravedad.isin(['moderada', 'grave'])

	mask_A = tiene_antecedente | es_moderada_o_grave
	ids = df['id_paciente'] if 'id_paciente' in df.columns else df.index
	set_A = set(ids[mask_A])

	print("\n=== Índice de compatibilidad ===")
	print(f"Tamaño de A (alergias o moderada/grave): {len(set_A)}")
	if len(set_A) == 0:
		print("No hay pacientes en A. No se puede calcular el índice.")
		return

	# Top N medicamentos
	medicamentos = df['medicamento'].fillna('Ninguno').astype(str)
	top_meds = medicamentos.value_counts().head(top_n).index.tolist()

	indices = []
	labels = []
	for med in top_meds:
		mask_B = medicamentos == med
		set_B = set(ids[mask_B])
		inter = set_A & set_B
		indice = len(inter) / len(set_A)
		indices.append(indice)
		labels.append(med)
		print(f"Medicamento: {med} — |A ∩ B| = {len(inter)} — Índice = {indice:.3f}")

	# Graficar índices
	plt.figure(figsize=(8,5))
	x = np.arange(len(labels))
	plt.bar(x, indices, color='salmon')
	plt.xticks(x, labels, rotation=45, ha='right')
	plt.ylim(0, 1)
	plt.ylabel('Compatibilidad (|A ∩ B| / |A|)')
	plt.title('Índice de compatibilidad por medicamento (top {})'.format(top_n))
	plt.tight_layout()

	# Guardar la figura en la carpeta raíz del proyecto
	base_dir = os.path.dirname(__file__)
	path_indice = os.path.join(base_dir, f'indice_compatibilidad_top{top_n}_medicamentos.png')
	plt.savefig(path_indice, dpi=300, bbox_inches='tight')
	print(f"Gráfica de índice de compatibilidad guardada en: {path_indice}")


def main():
	# Ruta del CSV en la misma carpeta del script
	ruta = os.path.join(os.path.dirname(__file__), 'Data_Biocompatibilidad.csv')
	if not os.path.exists(ruta):
		print(f"No se encontró el archivo: {ruta}")
		sys.exit(1)

	# Cargamos los datos
	df = cargar_datos(ruta)

	# Mostramos estadísticas básicas en consola
	mostrar_estadisticas(df)

	# Gráfica 1: Diagrama de Venn (látex vs antibiótico)
	diagrama_venn_latex_antibiotico(df)

	# Gráfica 2: Barras top5 materiales
	grafica_barras_top5_materiales(df)

	# Gráfica 3: Índice de compatibilidad por medicamento (top 5)
	indice_compatibilidad_por_medicamento(df, top_n=5)

	# Mostramos todas las gráficas generadas
	print("\nMostrando las gráficas. Cierra las ventanas para terminar el programa.")
	plt.show()


if __name__ == '__main__':
	main()
