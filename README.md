# Made With ML MLOps Course

### Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```

### Directory
```bash
tagifai/
├── data.py       - data processing utilities
├── evaluate.py   - evaluation components
├── main.py       - training/optimization operations
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```
