# Lean serving image: ONNX Runtime + Streamlit only (no PyTorch).
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"
COPY app ./app
COPY models ./models
ENV MODEL_PATH=/app/models/model.onnx
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
