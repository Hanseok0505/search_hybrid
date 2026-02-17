@echo off
call C:\ProgramData\Anaconda3\condabin\conda.bat activate hybrid-search
cd /d C:\Users\hs\Search_codex
set SAMPLE_MODE=true
set REDIS_ENABLED=true
set REDIS_URL=redis://localhost:6379/0
set ELASTIC_ENABLED=true
set ELASTIC_URL=http://localhost:9200
set MILVUS_ENABLED=false
set GRAPH_ENABLED=false
uvicorn app.main:app --host 0.0.0.0 --port 8000
