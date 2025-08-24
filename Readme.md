# Week 4 — TokenHF ✅

**This repository contains the complete work for Week 4 (Days 11–15).**  
It focuses on Hugging Face tokenization (EN + AR) and the documentation/packaging that completes the Titanic project.

## What’s here
- `tokenizer_demo.py` – Day 11: BERT tokenization demo  
  - English: `bert-base-uncased`  
  - Arabic: `asafaya/bert-base-arabic`
- `.gitignore`, misc project files

## Where the model/API live (Week 3)
The trained Titanic model, FastAPI service, Docker image, and DVC-tracked artifacts are in the Week 3 repo:

- **Week3 repo:** `MFaresJA/Titanic`  
  - API & Docker docs: `Week3/README_API.md`  
  - Main project README: `Week3/README.md`

> TL;DR: **TokenHF = Week4**, **Titanic = Week3 (model + API + DVC)**.

## Run the tokenizer demo (local)
```powershell
# from your venv at the project root
.\.venv\Scripts\Activate.ps1
python -m pip install "transformers>=4.42,<5"
python tokenizer_demo.py

ou’ll see tokens, input IDs, and attention masks for single sentences and batches in English and Arabic.
```
Run the tokenizer demo inside the API Docker container (optional)
# container started from the Week3 repo/image
docker exec -it titanic-final python tokenizer_demo.py