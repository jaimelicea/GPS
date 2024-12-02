<<<<<<< HEAD
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
# Replicación y Aplicación del Proceso "Generar, Podar, Seleccionar" para la Generación de Contradiscurso en Español
"[Generate, Prune, Select: A Pipeline for Counterspeech Generation against Online Hate Speech](https://arxiv.org/pdf/2106.01625.pdf)" (ACL-IJCNLP Findings 2021). 

## Introduction
El objetivo de este informe es replicar el proceso mencionado y adaptarlo al idioma español. En esta adaptación, se realizaron los siguientes cambios:
• Se cambio el embedding para el VAE (Variational Autoencoder) por uno que sea adecuado para el español.
• El modelo CoLa (Corpus of Linguistic Acceptability) se sustituyo por EsCoLa, una versión adaptada para evaluar la gramática en español.
• Se utilizará un codificador de oraciones multilingüe (sentence encoder multilingual) para asegurar que el modelo pueda trabajar eficazmente con enunciados en español.

El articulo propone un proceso en tres módulos diseñado para generar respuestas de contradiscurso eficaces y relevantes con el fin de contrarrestar el discurso de odio en línea:
• Generar: Utilizan un modelo generativo para producir una variedad de respuestas posibles al discurso de odio. Este paso promueve la diversidad de respuestas.
• Podar: Emplean un modelo BERT para filtrar las respuestas generadas que no sean gramaticalmente correctas o que no tengan sentido. Este filtro ayuda a eliminar las respuestas menos útiles.
• Seleccionar: Desarrollan un método novedoso basado en recuperación para seleccionar la respuesta de contradiscurso más relevante y adecuada de entre las respuestas generadas previamente. Este método asegura que la respuesta elegida sea pertinente y efectiva.

## Requirements
The code is based on Python 3.7. Please install the dependencies as below:  
```
pip install -r requirements.txt
```

En github solo estan los codigos y backups de reducido tamano. Todos los archivos para la replica estan guardados en el link siguiente: 
https://1drv.ms/f/s!AqL-CBysVDyCmKRG44J67hikS5J20w?e=CaG3f5

## Code:
Correr en siguiente orden:

### BERT_Fine_Tuning_esCoLA

### create_data_for_VAE_SP.ipynb (o en Main_JL_VAE_EN ingles): 
Creacion de archivo: conan_for_VAE.txt. Este es el input principal para entrenar el VAE. 
No necesario correr pues ya esta creado el archivo en github: GPS/data/conansp_for_VAE.txt (GPS/data/conan_for_VAE.txt)

### Main_JL_VAE_SP.ipynb (o en Main_JL_VAE_EN ingles):
Entrena el modelo Generar candidatos a partir de VAE: GPS/data/conansp_candidates.txt (GPS/data/conan_candidates.txt)


### Main_JL_Module_2y3_SP.ipynb (o en Main_JL_Module_2y3_EN ingles):
Poda gramaticalmente los candidatos y selecciona la mejor respuesta.
No necesario correr pues ya esta creado el archivo en github:  GPS/data/extracted_good_candidates_by_LQ_SP.pkl (GPS/data/extracted_good_candidates_by_LQ.pkl)

### JL_Results_Metric.ipynb:
Generar todas metricas de los resultados generados


