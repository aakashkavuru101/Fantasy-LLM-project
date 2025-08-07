"""Health check utilities for Fantasy LLM."""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import psutil
import torch
import httpx

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


def check_gpu_health() -> Dict[str, Any]:
    """Check GPU health and memory."""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "current_device": None,
    }
    
    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["current_device"] = torch.cuda.current_device()
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "allocated_memory": torch.cuda.memory_allocated(i) if i == torch.cuda.current_device() else 0,
                "reserved_memory": torch.cuda.memory_reserved(i) if i == torch.cuda.current_device() else 0,
            }
            gpu_info["devices"].append(device_info)
    
    return gpu_info


def check_disk_space() -> Dict[str, Any]:
    """Check disk space for model storage."""
    disk_info = {}
    
    # Check main directories
    directories = [
        settings.adapter_dir,
        settings.token_dir,
        settings.output_dir,
        settings.torch_home,
        settings.hf_home,
        Path("logs"),
    ]
    
    for directory in directories:
        if directory.exists():
            usage = psutil.disk_usage(str(directory))
            disk_info[str(directory)] = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": (usage.used / usage.total) * 100,
            }
    
    return disk_info


def check_memory() -> Dict[str, Any]:
    """Check system memory."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        "virtual": {
            "total": memory.total,
            "used": memory.used,
            "available": memory.available,
            "percent": memory.percent,
        },
        "swap": {
            "total": swap.total,
            "used": swap.used,
            "free": swap.free,
            "percent": swap.percent,
        },
    }


def check_model_files() -> Dict[str, Any]:
    """Check if model files exist."""
    files_info = {
        "adapter_exists": settings.adapter_dir.exists(),
        "tokenized_data_exists": settings.token_dir.exists(),
        "output_dir_exists": settings.output_dir.exists(),
    }
    
    # Check for specific model files
    model_files = [
        settings.adapter_dir / "adapter_config.json",
        settings.adapter_dir / "adapter_model.bin",
        settings.adapter_dir / "tokenizer_config.json",
    ]
    
    files_info["model_files"] = {}
    for file_path in model_files:
        files_info["model_files"][file_path.name] = file_path.exists()
    
    return files_info


def check_api_health() -> Dict[str, Any]:
    """Check API health if running."""
    api_info = {
        "reachable": False,
        "response_time": None,
        "status_code": None,
    }
    
    try:
        start_time = time.time()
        with httpx.Client() as client:
            response = client.get(
                f"http://{settings.api_host}:{settings.api_port}/health",
                timeout=10.0
            )
            api_info["reachable"] = True
            api_info["response_time"] = time.time() - start_time
            api_info["status_code"] = response.status_code
            api_info["response"] = response.json()
    except Exception as e:
        api_info["error"] = str(e)
    
    return api_info


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "boot_time": psutil.boot_time(),
        "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
    }


def comprehensive_health_check() -> Dict[str, Any]:
    """Run comprehensive health check."""
    logger.info("Running comprehensive health check")
    
    health_data = {
        "timestamp": time.time(),
        "system": get_system_info(),
        "memory": check_memory(),
        "gpu": check_gpu_health(),
        "disk": check_disk_space(),
        "model_files": check_model_files(),
        "api": check_api_health(),
        "settings": {
            "environment": settings.environment,
            "base_model": settings.base_model,
            "adapter_dir": str(settings.adapter_dir),
        },
    }
    
    return health_data


def main():
    """Main health check entry point."""
    health_data = comprehensive_health_check()
    
    print(json.dumps(health_data, indent=2))
    
    # Check for critical issues
    critical_issues = []
    
    # Check GPU availability if expected
    if not health_data["gpu"]["available"]:
        logger.warning("GPU not available")
        critical_issues.append("GPU not available")
    
    # Check memory usage
    if health_data["memory"]["virtual"]["percent"] > 90:
        critical_issues.append("High memory usage")
    
    # Check disk space
    for path, disk_info in health_data["disk"].items():
        if disk_info["percent"] > 90:
            critical_issues.append(f"High disk usage in {path}")
    
    # Check model files
    if not health_data["model_files"]["adapter_exists"]:
        critical_issues.append("Model adapter not found")
    
    if critical_issues:
        logger.error("Critical health issues found", issues=critical_issues)
        sys.exit(1)
    else:
        logger.info("Health check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()