"""Production-ready FastAPI application for Fantasy LLM."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .config import settings
from .logging import get_logger, log_inference, log_error

# Metrics
REQUEST_COUNT = Counter(
    "fantasy_llm_requests_total", "Total requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "fantasy_llm_request_duration_seconds", "Request duration"
)
INFERENCE_COUNT = Counter("fantasy_llm_inferences_total", "Total inferences")
INFERENCE_DURATION = Histogram(
    "fantasy_llm_inference_duration_seconds", "Inference duration"
)

logger = get_logger(__name__)

# Global model storage
model_cache = {}


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="Input prompt for story generation")
    max_new_tokens: Optional[int] = Field(
        default=None, ge=1, le=1024, description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.1, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.1, le=1.0, description="Top-p sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100, description="Top-k sampling parameter"
    )
    do_sample: bool = Field(default=True, description="Whether to use sampling")


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    generated_text: str = Field(..., description="Generated fantasy story")
    prompt_tokens: int = Field(..., description="Number of tokens in prompt")
    generated_tokens: int = Field(..., description="Number of tokens generated")
    total_tokens: int = Field(..., description="Total tokens processed")
    inference_time: float = Field(..., description="Inference time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]


async def load_model():
    """Load the model and tokenizer."""
    try:
        logger.info("Loading model", model=settings.base_model)
        
        tokenizer = AutoTokenizer.from_pretrained(str(settings.adapter_dir))
        model = AutoModelForCausalLM.from_pretrained(
            settings.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            return_full_text=False
        )
        
        model_cache["pipeline"] = model_pipeline
        model_cache["tokenizer"] = tokenizer
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Fantasy LLM API")
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down Fantasy LLM API")
    model_cache.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Fantasy LLM API",
    description="Production-ready API for LoRA fine-tuned Llama-2 fantasy story generation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment != "production" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer() if settings.api_key else None


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key if configured."""
    if settings.api_key and credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


def get_model_pipeline():
    """Get model pipeline from cache."""
    if "pipeline" not in model_cache:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return model_cache["pipeline"]


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time header and metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(process_time)
    
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    memory_usage = {}
    
    if gpu_available:
        memory_usage["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3
        memory_usage["gpu_reserved"] = torch.cuda.memory_reserved() / 1024**3
    
    return HealthResponse(
        status="healthy",
        model_loaded="pipeline" in model_cache,
        gpu_available=gpu_available,
        memory_usage=memory_usage
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_story(
    request: GenerateRequest,
    _: HTTPAuthorizationCredentials = Depends(verify_api_key) if settings.api_key else None
):
    """Generate a fantasy story."""
    start_time = time.time()
    
    try:
        pipeline_obj = get_model_pipeline()
        tokenizer = model_cache["tokenizer"]
        
        # Use request parameters or fallback to settings
        generation_params = {
            "max_new_tokens": request.max_new_tokens or settings.max_new_tokens,
            "temperature": request.temperature or settings.temperature,
            "top_p": request.top_p or settings.top_p,
            "top_k": request.top_k or settings.top_k,
            "do_sample": request.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # Generate text
        result = pipeline_obj(request.prompt, **generation_params)
        generated_text = result[0]["generated_text"]
        
        # Calculate token counts
        prompt_tokens = len(tokenizer.encode(request.prompt))
        generated_tokens = len(tokenizer.encode(generated_text))
        total_tokens = prompt_tokens + generated_tokens
        
        inference_time = time.time() - start_time
        
        # Update metrics
        INFERENCE_COUNT.inc()
        INFERENCE_DURATION.observe(inference_time)
        
        # Log inference
        log_inference(
            request.prompt,
            generated_text,
            inference_time=inference_time,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            inference_time=inference_time,
        )
        
    except Exception as e:
        log_error(e, {"prompt": request.prompt, "params": generation_params})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate story"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not settings.enable_metrics:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fantasy LLM API",
        "version": "1.0.0",
        "model": settings.base_model,
        "docs": "/docs" if settings.environment != "production" else "disabled",
    }


def main():
    """Main entry point for the API server."""
    import uvicorn
    
    uvicorn.run(
        "fantasyllm.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
        access_log=settings.api_access_log,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()