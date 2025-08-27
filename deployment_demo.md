
# Demo Deployment (single GPU host)
Run all services locally on a GPU machine; private data stays in ./data/private.

## Prereqs
- NVIDIA GPU + recent drivers (CUDA 12.x)
- Docker + NVIDIA Container Toolkit (`nvidia-ctk`)

## Run
```bash
cd demo_deploy
docker compose up --build
# UI  -> http://localhost:8501
# API -> http://localhost:8000
# cuOpt -> http://localhost:5000
```

## Secure sharing for stakeholders
- Share via Cloudflare Tunnel / Tailscale / reverse proxy with SSO.
- Keep ./data/private mounted on the GPU host; never upload data to public repos or the UI host.
