FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/

# The trained model artifact is gitignored (never committed to git — see
# CLAUDE.md), so a fresh clone of this repo (e.g. a hosting platform
# building straight from GitHub) has an empty models/ dir. Fall back to the
# published release asset in that case; a local build that already has a
# pre-trained artifact on disk skips this and never touches the network.
RUN if [ ! -f models/model_class_weighted.pkl ]; then \
        python -c "import urllib.request; urllib.request.urlretrieve( \
            'https://github.com/liamhavers/fraud-detection-system/releases/download/model-v1/model_class_weighted.pkl', \
            'models/model_class_weighted.pkl')"; \
    fi

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
