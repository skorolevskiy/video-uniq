"""
FastAPI Microservice for Video Uniqueness Generation
Modern async API with automatic documentation
"""

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uuid
import asyncio
from pathlib import Path
import shutil
import zipfile
import aiofiles
from video_uniquifier import VideoUniquifier

app = FastAPI(
    title="Video Uniquifier API",
    description="Generate unique video variations to bypass duplication detection",
    version="1.0.0"
)

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Job storage
jobs = {}


class JobStatus(BaseModel):
    job_id: str
    status: str
    input_file: str
    num_variations: int
    variations: List[dict] = []
    error: Optional[str] = None


class JobInfo(BaseModel):
    job_id: str
    status: str
    input_file: Optional[str] = None
    num_variations: Optional[int] = None


def create_zip_file(output_dir: str, job_id: str) -> Path:
    """Create a zip file from the output directory"""
    zip_path = Path(output_dir) / f'variations_{job_id}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in Path(output_dir).glob('*.mp4'):
            zipf.write(file, file.name)
        log_file = Path(output_dir) / 'variations_log.json'
        if log_file.exists():
            zipf.write(log_file, log_file.name)
    return zip_path


async def process_video_task(job_id: str, input_path: str, output_dir: str, num_variations: int):
    """Background task for video processing"""
    try:
        jobs[job_id]['status'] = 'processing'
        
        uniquifier = VideoUniquifier(
            input_video=input_path,
            output_dir=output_dir,
            num_variations=num_variations
        )
        await uniquifier.generate_variations()
        
        # Create zip file
        loop = asyncio.get_running_loop()
        zip_path = await loop.run_in_executor(None, create_zip_file, output_dir, job_id)
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['zip_file'] = str(zip_path)
        jobs[job_id]['variations'] = uniquifier.variations_log
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Video Uniquifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/upload", response_model=dict, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    num_variations: int = Form(default=10)
):
    """
    Upload a video and start processing variations
    
    - **video**: Video file (MP4, MOV, AVI, MKV)
    - **num_variations**: Number of variations to generate (1-50)
    """
    
    # Validate file extension
    file_ext = video.filename.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validate num_variations
    if num_variations < 1 or num_variations > 50:
        raise HTTPException(
            status_code=400,
            detail="num_variations must be between 1 and 50"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_path = UPLOAD_FOLDER / f"{job_id}_{video.filename}"
    async with aiofiles.open(input_path, 'wb') as out_file:
        while content := await video.read(1024 * 1024):  # Read in 1MB chunks
            await out_file.write(content)
    
    # Create output directory
    output_dir = OUTPUT_FOLDER / job_id
    output_dir.mkdir(exist_ok=True)
    
    # Initialize job
    jobs[job_id] = {
        'status': 'queued',
        'input_file': video.filename,
        'num_variations': num_variations,
        'variations': []
    }
    
    # Start background processing
    background_tasks.add_task(
        process_video_task,
        job_id,
        str(input_path),
        str(output_dir),
        num_variations
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video uploaded successfully. Processing started."
    }


@app.get("/api/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    """Get processing status for a specific job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/api/download/{job_id}")
def download_variations(job_id: str):
    """Download all variations as a ZIP file"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    zip_path = job['zip_file']
    
    if not Path(zip_path).exists():
        raise HTTPException(status_code=404, detail="Variations file not found")
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f'video_variations_{job_id}.zip'
    )


@app.get("/api/jobs", response_model=List[JobInfo])
def list_jobs():
    """List all jobs"""
    
    jobs_list = []
    for job_id, job_info in jobs.items():
        jobs_list.append(JobInfo(
            job_id=job_id,
            status=job_info['status'],
            input_file=job_info.get('input_file'),
            num_variations=job_info.get('num_variations')
        ))
    
    return jobs_list


def _cleanup_files(job_id: str):
    # Remove output directory
    output_dir = OUTPUT_FOLDER / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Remove uploaded file
    for file in UPLOAD_FOLDER.glob(f"{job_id}_*"):
        file.unlink()


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Delete job files and cleanup resources"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _cleanup_files, job_id)
        
        # Remove from jobs dict
        del jobs[job_id]
        
        return {"message": "Job cleaned up successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Video Uniquifier FastAPI Server...")
    print("📝 API Documentation: http://localhost:8000/docs")
    print("🔧 Alternative docs: http://localhost:8000/redoc")
    print("\n🌐 Server starting on http://localhost:8000\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
