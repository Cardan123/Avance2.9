# MultimodalRAG Pipeline
## Embeddings Section
## General Description

This project implements a multimodal pipeline to segment images, extract text using OCR, vectorize each segment using AI models (SAM and CLIP), and store the embeddings in a MongoDB Atlas database. Additionally, we now have an available MCP Server that enables data ingestion and offers interoperability with other systems.

The pipeline supports two modes of operation:
- **Automatic segmentation:** Uses SAM to detect and crop image segments.
- **Pre-segmented images:** You can provide pre-segmented images (with their bounding boxes and labels), and the pipeline will skip the automatic segmentation step, directly processing the segments and their metadata.

Key features:
- Automatic image segmentation with SAM (Segment Anything Model).
- Text extraction using OCR with Tesseract.
- Vectorization of each segment with CLIP (Contrastive Language-Image Pretraining).
- Manual/automatic classification of segments.
- Storage of results in MongoDB Atlas.
- Visualization and saving of detected segments.

> **Note:** The path to the Tesseract executable is configured in the `config_ingest.yaml` file under the `tesseract_path` key. Ensure this path is correct for OCR to work properly.

# Table of Contents
- [General Description](#general-description)
- [Pipeline Flow Diagram](#pipeline-flow-diagram)
- [Environment Setup](#environment-setup)
- [YAML Configuration](#yaml-configuration)
- [Model Download](#model-download)
- [MCP Server for Data Ingestion](#mcp-server-for-data-ingestion)
- [Pipeline Execution](#pipeline-execution)
- [Usage Example](#usage-example)
	- [Process Images with Automatic Segmentation](#1-process-images-with-automatic-segmentation)
	- [Process Pre-Segmented Images](#2-process-pre-segmented-images)
- [Recommended Tool for Manual Labeling](#recommended-tool-for-manual-labeling)
- [segmented_images_dataset.json File](#segmented_images_datasetjson-file)
- [Behavior if the File Does Not Exist](#behavior-if-the-file-does-not-exist)
- [Location of Generated Segments](#location-of-generated-segments)
- [Document Generation for MongoDB](#document-generation-for-mongodb)
- [Template for .env File](#template-for-env-file)
- [Notes](#notes)

## Pipeline Flow Diagram

```
┌───────────────┐
│  Images       │
└─────┬────────┘
      │
      ▼
┌─────────────────────────────┐
│  Automatic Segmentation     │◄─────────────┐
│  (SAM)                      │              │
└─────────────┬──────────────┘              │
              │                             │
              ▼                             │
┌─────────────────────────────┐              │
│  Pre-Segmented Images       │──────────────┘
│  (manual or preprocessed)   │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Vectorization (CLIP)       │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Classification             │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Storage in MongoDB         │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Visualization/Export       │
└─────────────────────────────┘
```

## Environment Setup

To use the pipeline, ensure you have the latest version of Tesseract OCR installed. You can download it from the following link:

[Tesseract OCR 5.5.0 for Windows (64 bits)](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)

Once installed, configure the path to the executable in the `config_ingest.yaml` file under the `tesseract_path` key.

## YAML Configuration

The pipeline uses a YAML configuration file (`config_ingest.yaml`) where you can set:

- The path to the image folder (`image_folder`).
- The path to the manual segments file (`segmented_images_dataset`).
- The SAM segmentation mode (`sam_mode`), which can be `manual` or `auto`.
- The path to the Tesseract executable (`tesseract_path`).

Configuration example:

```yaml
image_folder: data/images
segmented_images_dataset: data/images/segments/segmented_images_dataset.json
sam_mode: manual  # Can be 'manual' or 'auto'. Adjust the SAM segmentation mode here.
tesseract_path: "C:\\Users\\user\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"  # Path to the Tesseract executable
RECREATE_VECTOR_DB: False  # If True, the pipeline recreates the vector database on each run. If False, skips recreation.

> **Usage of RECREATE_VECTOR_DB:**
> This parameter controls whether the pipeline should recreate the vector database on each run. If set to `True`, all images will be processed and the database will be overwritten. If left as `False`, the pipeline will skip recreation and only process new images.
RECREATE_VECTOR_DB: False  # Can be True or False. Controls whether the vector database is recreated.
```

You can modify these paths and the segmentation mode according to your project structure and needs.
1. Clone the repository to the suggested path:
	```bash
	cd C:\Git
	git clone https://github.com/edwinhdez/MultimodalRAG.git
	cd MultimodalRAG
	```
2. Create and activate a virtual environment (recommended):
	```bash
	python -m venv venv
	# On Windows:
	venv\Scripts\activate
	# On Linux/Mac:
	source venv/bin/activate
	```
3. Install the dependencies:
	```bash
	pip install -r requirements.txt
	```

## Model Download
To avoid downloading the models from scratch on each run, you can manually download them and save them in the following folders:

- **SAM (Segment Anything Model):**
  - URL: https://github.com/facebookresearch/segment-anything
  - Download the checkpoint `sam_vit_b_01ec64.pth` and save it to:
	 ```
	 src/models/SAM/sam_vit_b_01ec64.pth
	 ```

- **CLIP (OpenAI CLIP):**
  - URL: https://huggingface.co/openai/clip-vit-base-patch16
	- The model is automatically downloaded on the first attempt using Hugging Face. On subsequent runs, the pipeline loads it directly from the cached directory:
		 ```
		 src/models/CLIP/openai_clip-vit-base-patch16/models--openai--clip-vit-base-patch16/snapshots/<snapshot_id>/
		 ```
		 (Ensure that the files `pytorch_model.bin` and `config.json` exist in the snapshot)

## MCP Server for Data Ingestion

The project now includes an available MCP Server for data ingestion. This server allows efficient data management and processing within the pipeline.

### MCP Server Configuration in Claude Desktop

The configuration file for Claude Desktop is located at:

```
C:\Git\MultimodalRAG\src\mcpserver\claude_desktop_config\config.json
```

The content of the file is as follows:

```json
{
  "mcpServers": {
    "IngestPipelineMCPServer": {
      "command": "C:\\Git\\MultimodalRAG\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\Git\\MultimodalRAG\\src\\mcpserver\\ingest_pipeline_mcp_server.py"
      ]
    }
  }
}
```
## Pipeline Execution
1. Configure the `.env` file (see template below).
2. Run the main pipeline:
	```bash
	python src/main.py
	```

## Usage Example

### 1. Process Images with Automatic Segmentation

Place your images in the configured folder (default `data/images/`). The pipeline will automatically detect the segments:

```bash
python src/main.py --image_folder data/images/
```

### 2. Process Pre-Segmented Images

#### Recommended Tool for Manual Labeling

You can use [VoTT](https://github.com/microsoft/VoTT/releases/download/v2.2.0/vott-2.2.0-win32.exe) to visually and easily set the tags and bounding boxes for each image. VoTT allows exporting the data in JSON format compatible with the pipeline.

# segmented_images_dataset.json File

You can manually define the segments in the `segmented_images_dataset.json` file. This file allows you to predefine the segments of the images for the ingestion pipeline. Its purpose is to provide pre-segmented information including the image name, label (`tags`), bounding box (`boundingBox`), and segment points.

**Location:**

The file should be located at:

`C:\Git\MultimodalRAG\data\images\segments\segmented_images_dataset.json`

**Structure:**

The file is a JSON array of objects, each representing a segment:

```json
[
	{
		"id": "pig_2.jpg",
		"type": "RECTANGLE",
		"tags": ["Adapter"],
		"boundingBox": {
			"height": 811.135,
			"width": 1013.10,
			"left": 922.48,
			"top": 0
		},
		"points": [
			{"x": 922.48, "y": 0},
			{"x": 1935.58, "y": 0},
			{"x": 1935.58, "y": 811.13},
			{"x": 922.48, "y": 811.13}
		]
	},
	{
		"id": "pig_2.jpg",
		"type": "RECTANGLE",
		"tags": ["Pig"],
		"boundingBox": {
			"height": 858.07,
			"width": 1013.10,
			"left": 453.05,
			"top": 638.64
		},
		"points": [
			{"x": 453.05, "y": 638.64},
			{"x": 1466.15, "y": 638.64},
			{"x": 1466.15, "y": 1496.72},
			{"x": 453.05, "y": 1496.72}
		]
	}
]
```

# Behavior if the File Does Not Exist

If the `segmented_images_dataset.json` file is not present, the pipeline will perform automatic segment detection on the images using the configured model (default SAM).

# Location of Generated Segments

All generated segments (whether automatic or manual) are saved in the folder:

`C:\Git\MultimodalRAG\data\images\segments`


## Document Generation for MongoDB

For each processed segment, the pipeline generates a document containing:

- The embedding generated by the CLIP model.
- The path of the original image.
- The path of the generated segment.
- The label (`label`) of the segment.
- The bounding box (`bbox`) of the segment.
- The text extracted using OCR (`text`).
- The coordinates of the detected texts (`ocr_bboxes`).

Example document structure:

```json
{
	"image_id": "data/images/pig_2.jpg",
	"image_embedding": [0.123, 0.456, ...],
	"image_path": "data/images/pig_2.jpg",
	"segment_path": "data/images/segments/pig_2/segment_0.png",
	"label": "Pig",
	"bbox": [453.05, 638.64, 1466.15, 1496.72],
	"text": "Texto extraído",
	"ocr_bboxes": [
		{"bbox": [460, 650, 500, 670], "text": "Hello"},
		{"bbox": [510, 680, 550, 700], "text": "World"}
	]
}
```

This document is stored in the configured MongoDB Atlas collection. The embedding and storage process is designed to be flexible and can be adjusted according to the project's needs.

## Template for .env File
```
MONGODB_ATLAS_URI=mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_ATLAS_DB=MultimodalRAG
MONGODB_ATLAS_COLLECTION=ImageEmbeddings
HUGGINGFACE_TOKEN=tu_token_huggingface
```

> **Note:** The `RECREATE_VECTOR_DB` parameter is now configured in the `config_ingest.yaml` file and not in the `.env` file.

## Notes
- If the models are already downloaded in the indicated folders, the pipeline will use them directly and avoid unnecessary downloads.
- To obtain the Hugging Face token, go to https://huggingface.co/settings/tokens
- You can adjust the image folder in the YAML configuration file or in the class parameter.



