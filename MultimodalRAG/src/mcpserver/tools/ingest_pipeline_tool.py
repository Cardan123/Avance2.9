from ingest.image_ingest_pipeline import ImageIngestPipeline

def run_ingest_pipeline():
    pipeline = ImageIngestPipeline()
    pipeline.run()
    return {"status": "success", "message": "Image pipeline ran successfully."}