#!/bin/bash
cd habla
export RECORDING_ENABLED=1
export HF_TOKEN=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
/c/Python312/python -m uvicorn server.main:app --host 0.0.0.0 --port 8002
