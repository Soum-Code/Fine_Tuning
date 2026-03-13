from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.inference.model_deployer import ModelDeployer

app = FastAPI(title="Industrial LoRA Fine-tuned Model API")

# Global model instance
model_deployer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    generated_text: str

@app.on_event("startup")
async def load_model():
    global model_deployer
    # Note: args.model_path and args.device need to be accessible
    # We'll use a trick/global or just parse here if uvicorn allows
    pass

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if model_deployer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = model_deployer.generate_text(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return GenerationResponse(generated_text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def start_server():
    global model_deployer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    model_deployer = ModelDeployer(args.model_path, args.device)
    model_deployer.load_model()

    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    start_server()
