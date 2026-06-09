# backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from .model import generator # Import the global generator instance
from .schema import PatientSummary, GenerationResponse, GenerationRequest # Import schemas
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedReportGen AI API", description="API for generating clinical notes.")

# --- Serve static files from the 'dist' directory ---
# Assuming your built frontend is in D:\VScodefiles\MedReportGen-AI\frontend\dist
# You might need to adjust the path if your structure is different
frontend_dist_path = r"D:\VScodefiles\MedReportGen-AI\frontend\dist"
print(f"Serving static files from: {frontend_dist_path}") # Debug log
if os.path.exists(frontend_dist_path):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_dist_path, "assets")), name="static")
    # Serve index.html for the root path and any other path not matching API routes
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Try to serve the requested path from the dist folder
        requested_file = os.path.join(frontend_dist_path, full_path)
        if os.path.isfile(requested_file):
            return FileResponse(requested_file)
        # If the path doesn't match a file, assume it's a frontend route (e.g., /dashboard)
        # and serve the main index.html file (SPA routing)
        index_path = os.path.join(frontend_dist_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            # If index.html doesn't exist, the frontend build might be missing
            logger.error(f"Frontend build not found at {index_path}")
            return {"error": "Frontend build not found. Run 'npm run build' in the frontend directory."}
else:
    logger.error(f"Frontend build directory does not exist: {frontend_dist_path}")
    # You might want to raise an error or handle this differently
    # For now, we'll just log it and proceed without serving the frontend

# --- CORS Middleware (Still needed if you plan to access the API externally later) ---
# If you *only* serve the frontend from here, you might not strictly need CORS for internal frontend->API calls,
# but it's good practice to have it configured if the API might be used by other clients later.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust if you want to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Middleware ---

class HealthCheck(BaseModel):
    status: str = "OK"

@app.get("/", response_class=FileResponse)
async def read_root():
    # Serve the built index.html for the root path
    index_path = os.path.join(frontend_dist_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.error(f"Frontend index.html not found at {index_path}")
        return {"error": "Frontend index.html not found. Run 'npm run build' in the frontend directory."}

@app.get("/health", response_model=HealthCheck)
def health_check():
    # Could add more sophisticated checks here (e.g., model loaded, GPU accessible)
    return HealthCheck(status="OK")

@app.post("/generate", response_model=GenerationResponse)
def generate_clinical_note(request: GenerationRequest):
    try:
        # Access the summary from the request body
        summary = request.summary

        # Log the received summary (optional, for debugging)
        logger.info(f"Received summary: {summary}")

        # Call the generator function
        generated_note = generator.generate_note(summary.dict())

        # Define some basic rules for warnings (example)
        warnings = []
        if summary.pain_intensity > 8:
            warnings.append("High pain intensity reported.")
        if summary.hemoglobin < 7.0:
            warnings.append("Critically low hemoglobin level.")
        if summary.oxygen_saturation < 90.0:
            warnings.append("Low oxygen saturation detected.")

        # Calculate a simple confidence score (placeholder - based on model output or other metrics)
        # For now, use a simple heuristic based on length and keywords, or just return 0.8 as a default
        # A real implementation might use model probabilities or other metrics
        confidence_score = 0.8

        # Prepare the response
        response = GenerationResponse(
            generated_note=generated_note,
            warnings=warnings,
            confidence_score=confidence_score
        )
        return response
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating clinical note: {str(e)}")

# --- Example endpoint for testing ---
@app.post("/test_generate")
def test_generate(summary: PatientSummary): # Use the schema directly for this endpoint if needed
    # This endpoint might be less necessary if /generate is the main one
    # It's kept here as an example if needed, but /generate is the primary one based on GenerationRequest
    try:
        logger.info(f"Test endpoint received summary: {summary}")
        generated_note = generator.generate_note(summary.dict())
        return {"generated_note": generated_note}
    except Exception as e:
        logger.error(f"Error during test generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in test generation: {str(e)}")

# --- Include other routers if you have them ---
# from .routers import some_router
# app.include_router(some_router, prefix="/some_prefix", tags=["some_tag"])

# --- End of main.py ---
