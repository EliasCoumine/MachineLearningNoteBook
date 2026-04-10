# Lab 2 – Aerial House Segmentation Pipeline
**CEG4195/SEG4180 Applied ML · uOttawa Winter 2026**

Enhances the Lab 1 sentiment-analysis container with:
- **Secrets injection** via `python-dotenv`
- **CI/CD** via GitHub Actions (test → build → push to Docker Hub)
- **UNet segmentation model** (ResNet-34 encoder, transfer learning) trained
  on aerial imagery to detect building footprints
- **Pixel mask generation** code from Week 7 (colour threshold + SAM)
- **Metrics**: IoU and Dice score

---

## Project Structure

```
LAB2ML/
├── app.py                          # Flask API (segmentation endpoint)
├── Dockerfile                      # Multi-stage build
├── docker-compose.yml              # Compose with .env support
├── requirements.txt
├── .env.example                    # Template – copy to .env and fill in
├── .gitignore
├── .github/workflows/ci_cd.yml     # GitHub Actions CI/CD
├── dataset/
│   ├── prepare_dataset.py          # Download + split dataset
│   └── pixel_mask_generation.py   # Week 7 mask generation (threshold / SAM)
├── model/
│   ├── unet.py                    # UNet architecture
│   ├── train.py                   # Training script
│   └── evaluate.py                # Evaluation (IoU, Dice, visualisations)
├── tests/
│   └── test_api.py                # Pytest unit tests
└── notebooks/
    └── segmentation_training.ipynb
```

---

## Quick Start

### 1. Clone and configure secrets
```bash
git clone <your-repo>
cd LAB2ML
cp .env.example .env
# Edit .env and fill in HF_TOKEN, DOCKERHUB_USERNAME, etc.
```

### 2. Prepare the dataset
```bash
python dataset/prepare_dataset.py --output_dir data
```

### 3. Generate pixel masks (Week 7 code)
```bash
# Colour-threshold approach (no GPU needed)
python dataset/pixel_mask_generation.py \
    --input_dir  data/train/images \
    --output_dir data/train/generated_masks \
    --method     threshold

# SAM approach (requires SAM checkpoint)
python dataset/pixel_mask_generation.py \
    --input_dir      data/train/images \
    --output_dir     data/train/generated_masks \
    --method         sam \
    --sam_checkpoint /path/to/sam_vit_b_01ec64.pth
```

### 4. Train the segmentation model
```bash
python model/train.py \
    --data_dir data --ckpt_dir checkpoints \
    --epochs 30 --batch_size 8 --lr 1e-4
```

### 5. Evaluate on the test set
```bash
python model/evaluate.py \
    --ckpt_path checkpoints/best_model.pth \
    --out_dir   results
```

### 6. Run the API locally
```bash
docker compose up --build
curl http://localhost:5001/
curl -X POST http://localhost:5001/segment \
     -F "file=@data/test/images/00001.png"
```

### 7. Run tests
```bash
pytest tests/ -v
```

---

## CI/CD Pipeline (GitHub Actions)

| Job | Trigger | Description |
|-----|---------|-------------|
| **test** | all pushes / PRs | `pytest tests/` |
| **build** | after test passes | `docker buildx build` |
| **push** | push to `main` | push to Docker Hub |
| **smoke** | after push | `docker compose up` + health check |

**Required GitHub Secrets:**
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (Docker Hub Personal Access Token)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Health check |
| `POST` | `/segment` | Segment houses in an aerial image |

### `/segment` request formats

**Multipart (file upload):**
```bash
curl -X POST http://localhost:5001/segment \
     -F "file=@aerial.png"
```

**JSON (base64):**
```bash
curl -X POST http://localhost:5001/segment \
     -H "Content-Type: application/json" \
     -d '{"image_b64": "<base64-encoded-png>"}'
```

**Response:**
```json
{
  "mask_b64":   "<base64-encoded binary mask PNG>",
  "metrics":    null,
  "input_size": [512, 512],
  "output_size": [256, 256]
}
```

Optionally include `mask_b64` (ground-truth) in the request to receive IoU/Dice metrics:
```json
{
  "metrics": {"iou": 0.7823, "dice": 0.8381}
}
```

---

## Docker Hub
`docker pull eliasc2004/segmentation-api:latest`
