import logging
import logging.config
import json
import time
import uuid
import re
from typing import Optional, Any, Dict, Union, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel

from config import LOGS_DIR

# --- 1. Security Logic (Recursive Masking) ---
SENSITIVE_PATTERN = re.compile(r'(password|passwd|pwd|secret|token|api_key|apikey|email|e-mail)', re.IGNORECASE)

def mask_sensitive_data(data: Any) -> Any:
    """
    Recursively traverse dict/list to mask sensitive keys.
    This fixes the issue where Pydantic models weren't being masked properly.
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if SENSITIVE_PATTERN.search(str(k)):
                new_data[str(k)] = "***MASKED***"
            else:
                new_data[str(k)] = mask_sensitive_data(v)
        return new_data
    elif isinstance(data, list):
        return [mask_sensitive_data(item) for item in data]
    else:
        return data

class SensitiveDataFilter(logging.Filter):
    """Legacy text-based filter for standard string logs"""
    def __init__(self):
        super().__init__()
        self.patterns = [
            re.compile(r'(password|passwd|pwd|secret|token|api_key|apikey)=([^&"\s]+)', re.IGNORECASE),
            re.compile(r'(email|e-mail)=([^&"\s]+)', re.IGNORECASE)
        ]

    def filter(self, record):
        # Skip masking if it's a Pydantic object (handled by CustomJsonFormatter)
        if isinstance(record.msg, BaseModel):
            return True
            
        msg = str(record.msg)
        for pattern in self.patterns:
            msg = pattern.sub(r'\1=***MASKED***', msg)
        record.msg = msg
        return True

# --- 2. Data Class (Pydantic) ---
class RequestLogData(BaseModel):
    """Standardized structure for request/response logs."""
    request_id: str
    method: str
    path: str
    client_ip: str
    process_time_ms: float
    status_code: int
    user_agent: Optional[str] = None
    query_params: Optional[Dict[str, Any]] = None
    body: Optional[Union[Dict, List, str]] = None  # Added Body capture
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

# --- 3. Custom Formatter ---
class CustomJsonFormatter(logging.Formatter):
    """Formatter that handles Pydantic models and applies security masking"""
    def format(self, record):
        if isinstance(record.msg, BaseModel):
            # 1. Convert Pydantic object to dict
            log_obj = record.msg.model_dump()
            
            # 2. Apply Security Masking (Recursive)
            log_obj = mask_sensitive_data(log_obj)
            
            # 3. Add metadata
            log_obj['timestamp'] = self.formatTime(record)
            log_obj['level'] = record.levelname
            
            return json.dumps(log_obj)
        else:
            return super().format(record)

# --- 4. Configuration ---
def setup_logging(config_name="default"):
    """Load logging configuration"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "app.log"

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "sensitive_filter": {"()": SensitiveDataFilter}
        },
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            "json": {
                "()": CustomJsonFormatter,
                "format": "%(asctime)s %(levelname)s %(message)s" 
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
                "filters": ["sensitive_filter"]
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_file),
                "formatter": "json",
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
                "filters": ["sensitive_filter"]
            }
        },
        "loggers": {
            "": {"handlers": ["console", "file"], "level": "INFO", "propagate": True},
            "uvicorn.access": {"handlers": ["console"], "level": "WARNING", "propagate": False}
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("sales_forecasting")

# --- 5. Middleware ---
class APILoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("sales_forecasting.middleware")

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        
        # Capture Request Body
        # NOTE: await request.body() caches the body, so it's safe to read here
        request_body_content = None
        try:
            body_bytes = await request.body()
            if body_bytes:
                request_body_content = json.loads(body_bytes)
        except Exception:
            # If body is not JSON or empty, keep it None or log as string
            pass

        try:
            response = await call_next(request)
            
            # Process Response Body (Non-blocking)
            response_body = [chunk async for chunk in response.body_iterator]
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            
            process_time = (time.time() - start_time) * 1000
            
            log_data = RequestLogData(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
                status_code=response.status_code,
                process_time_ms=round(process_time, 2),
                user_agent=request.headers.get("user-agent"),
                query_params=dict(request.query_params),
                body=request_body_content  # Log captured body
            )
            
            self.logger.info(log_data)
            
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}", exc_info=True)
            raise e
