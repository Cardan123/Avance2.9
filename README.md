# Avance 2.9 — MultimodalRAG

Este repositorio contiene el avance del proyecto MultimodalRAG, una canalización (pipeline) multimodal para segmentar imágenes, extraer texto (OCR), vectorizar segmentos con modelos de visión/lingüísticos y almacenar embeddings en MongoDB Atlas. Incluye scripts de ingesta, configuración, notebooks y utilidades para correr la pipeline de punta a punta.

## Estructura del repositorio
- `Avance2.9.pdf`: Documento con el reporte/avance correspondiente.
- `MultimodalRAG/`: Código fuente principal del proyecto.
  - `README.md`: Guía detallada del pipeline (flujo, instalación, ejecución, configuración YAML, variables de entorno, ejemplos).
  - `src/`: Módulos del pipeline (ingesta, modelos, servidor MCP, main, helpers, etc.).
  - `data/`: Datos de ejemplo (imágenes y segmentos generados).
  - `config_ingest.yaml` y `config_retrieval.yaml`: Archivos de configuración.
  - `requirements.txt`: Dependencias de Python.
  - `notebooks/`: Cuadernos de experimentación.
  - `test/`: Pruebas.

> Nota: Dentro de `MultimodalRAG/` existe un `.git/`. Si deseas un único repositorio en la raíz (`Avance2.9/`), podemos remover ese `.git` interno o convertirlo en submódulo. Ver la sección “Control de versiones”.

## Requisitos
- Python 3.10+ recomendado
- Tesseract OCR instalado y accesible en el sistema
- Cuenta/URI de MongoDB Atlas (si utilizarás almacenamiento en la nube)
- (Opcional) Token de Hugging Face para acelerar descargas de modelos

## Configuración rápida
1. Crear entorno virtual e instalar dependencias (desde `MultimodalRAG/`):
   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate

   pip install -r requirements.txt
   ```
2. Configurar `config_ingest.yaml` (ruta de imágenes, ruta de Tesseract, modo SAM, etc.).
3. Crear archivo `.env` (ver plantilla en `MultimodalRAG/README.md`).
4. (Opcional) Descargar pesos de modelos según indica `MultimodalRAG/README.md`.

## Ejecución básica
Desde `MultimodalRAG/`:
```bash
python src/main.py
```

Ejemplo con carpeta de imágenes específica:
```bash
python src/main.py --image_folder data/images/
```

Revisa `MultimodalRAG/README.md` para:
- Flujo completo del pipeline
- Ejecución con segmentación automática vs. presegmentada
- Estructura de `segmented_images_dataset.json`
- Variables de entorno y configuración YAML
- Servidor MCP para ingestión

## Control de versiones (Git)
Hay dos opciones para estructurar el repositorio:
- Opción A — Un solo repo en la raíz (`Avance2.9/`):
  - Remover `MultimodalRAG/.git/` y versionar todo desde la raíz.
- Opción B — Mantener `MultimodalRAG/` como repo independiente o convertirlo en submódulo:
  - Si ya tiene remoto propio, podemos declararlo como submódulo del repo raíz.

Indica tu preferencia y procedo a configurarlo. Si eliges Opción A, inicializaré Git en la raíz y haré el primer commit con este README, el PDF y la carpeta `MultimodalRAG/`.

## Licencia
Define aquí la licencia del proyecto (por ejemplo, MIT, Apache-2.0). Si lo prefieres, puedo agregar un archivo `LICENSE`.
