FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/

# The served artifacts are a matched pair: the unweighted XGBoost model and
# the Platt calibrator fitted to that exact model on the validation slice.
# Both are gitignored (never committed to git — see CLAUDE.md), so a fresh
# clone of this repo (e.g. a hosting platform building straight from GitHub)
# has an empty models/ dir. In that case, or if either file of the pair is
# missing, download both from the published `model-v2` release, so the image
# never pairs a model with a calibrator from a different training run. A
# local build with both files on disk (`python -m src.models.train` writes
# them together) skips this and never touches the network.
ARG MODEL_RELEASE_URL=https://github.com/liamhavers/fraud-detection-system/releases/download/model-v2
RUN if [ ! -f models/model_unweighted.pkl ] || [ ! -f models/calibrator_unweighted_platt.pkl ]; then \
        for f in model_unweighted.pkl calibrator_unweighted_platt.pkl; do \
            python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
                "$MODEL_RELEASE_URL/$f" "models/$f" || exit 1; \
        done; \
    fi

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
