import os
import sys
import json
import time
import subprocess
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# Make sure the static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory database for rate limiting
# Format: { api_key: {"date": "YYYY-MM-DD", "count": 0} }
API_KEYS = {}
DAILY_LIMIT = 5

class GenerateRequest(BaseModel):
    api_key: str
    model: str
    prompt: str

@app.get("/")
def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/keygen")
def generate_key():
    # Generate a unique API key starting with 'sb_'
    new_key = f"sb_{uuid.uuid4().hex[:12]}"
    API_KEYS[new_key] = {"date": time.strftime("%Y-%m-%d"), "count": 0}
    return {"api_key": new_key}

@app.post("/generate")
def generate_text(req: GenerateRequest):
    # 1. API Key Validation
    if req.api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key. Kevin is disappointed.")
        
    today = time.strftime("%Y-%m-%d")
    user_data = API_KEYS[req.api_key]
    
    # Reset daily limit if it's a new day
    if user_data["date"] != today:
        user_data["date"] = today
        user_data["count"] = 0
        
    # 2. Rate Limiting Check
    if user_data["count"] >= DAILY_LIMIT:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({DAILY_LIMIT}/day). Pay your fees.")
        
    if req.model not in ["k1", "c1"]:
        raise HTTPException(status_code=400, detail="Invalid model. Choose k1 or c1.")
        
    # Increment usage counter
    user_data["count"] += 1
    
    # 3. Model Inference (using pure NumPy script)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(base_dir, "numpy_inference.py")
    model_dir = os.path.join(base_dir, req.model)
    
    try:
        # Run inference in a subprocess
        result = subprocess.run(
            [sys.executable, script_path, model_dir, req.prompt],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        output = result.stdout
        if "--- Output ---" in output:
            generated = output.split("--- Output ---")[1].split("--------------")[0].strip()
        else:
            generated = output.strip()
            
        return {"response": generated, "remaining": DAILY_LIMIT - user_data["count"]}
        
    except subprocess.CalledProcessError as e:
        print("Inference Error:", e.stderr)
        raise HTTPException(status_code=500, detail="Model inference failed. The servers are down again.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
