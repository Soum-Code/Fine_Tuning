# Deploy API server
python -m src.inference.api_server `
  --model-path ./checkpoints/final_model `
  --port 8000 `
  --device cpu
