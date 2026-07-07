# Insta-Forecast

**Author:** Praneeth

Insta-Forecast is an AI-powered Instagram engagement predictor. It extracts real-time metadata, performs NLP-driven feature engineering, and uses XGBoost regressors to predict likes, views, and comments, and to forecast 1–60 day growth curves.

Key features:
- Real-time metadata extraction using Instaloader
- NLP processing and TF-IDF / SVD embedding
- XGBoost regressors for likes, views, and comments
- FastAPI backend with simple JSON-based endpoints
- Minimal frontend for quick predictions

Repository structure:
- [app.py](app.py) — FastAPI backend API
- [index.html](index.html) — Frontend UI for predictions
- [instagram_scraper.py](instagram_scraper.py) — Scrapes Instagram metadata
- [process.py](process.py) — Preprocessing and feature engineering
- [cleaned.csv](cleaned.csv) — Raw/cleaned dataset
- [processed.csv](processed.csv) — Processed dataset used for modeling
- [processed1.csv](processed1.csv) — Alternate processed dataset
- [train_likes.py](train_likes.py) — Trains likes model
- [train_views.py](train_views.py) — Trains views model
- [train_comments.py](train_comments.py) — Trains comments model
- [models/likes_model_dayall.joblib](models/likes_model_dayall.joblib) — Trained likes model
- [models/views_model_dayall.joblib](models/views_model_dayall.joblib) — Trained views model
- [models/comments_model_dayall.joblib](models/comments_model_dayall.joblib) — Trained comments model

Project goals
- Quickly predict engagement metrics (likes, views, comments) for new Instagram posts
- Produce a 1–60 day growth curve for post engagement
- Provide a backend API and a minimal frontend UI for interacting with the models
- Offer an extensible pipeline for model retraining and feature engineering

Quick start

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Start backend
```bash
python app.py
```

3) Open web UI
Open [index.html](index.html) or visit `http://127.0.0.1:8000` to use the minimal frontend.

API Endpoints
- POST `/predict` — Predict likes, views, comments for a given post
  - Example:
    ```bash
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"post_url":"https://www.instagram.com/p/POST_ID/"}'
    ```
- POST `/predict_curve` — Predict 1–60 day growth curve
  - Example:
    ```bash
    curl -X POST http://127.0.0.1:8000/predict_curve \
      -H "Content-Type: application/json" \
      -d '{"post_url":"https://www.instagram.com/p/POST_ID/"}'
    ```

How it works (overview)
1. [instagram_scraper.py](instagram_scraper.py) extracts a post's metadata: caption, hashtags, timestamp, likes, comments, video views, and thumbnail.
2. [process.py](process.py) or preprocessing steps tokenize/clean captions, extract features (hashtag counts, SEO metrics, temporal features), compute sentiment, and vectorize text using TF-IDF + SVD embeddings.
3. Models are trained using [train_likes.py](train_likes.py), [train_views.py](train_views.py), and [train_comments.py](train_comments.py). Trained models are stored in [models/](models/).
4. [app.py](app.py) loads saved models and returns predictions via the REST API.

Training and evaluation
- Use the provided preprocessing and training scripts to retrain models on your data.
- Example:
  ```bash
  python train_likes.py
  python train_views.py
  python train_comments.py
  ```
- Current model metrics from sample dataset:
  - Likes: RMSE 24105.43, MAE 4988.51, $R^2$ 0.99808
  - Views: RMSE 65115.18, MAE 15848.04, $R^2$ 0.99898
  - Comments: RMSE 582.57, MAE 49.55, $R^2$ 0.99280

Data
- `cleaned.csv` — cleaned raw dataset used for feature extraction
- `processed.csv`, `processed1.csv` — processed datasets used during training

Model files
- [models/likes_model_dayall.joblib](models/likes_model_dayall.joblib)
- [models/views_model_dayall.joblib](models/views_model_dayall.joblib)
- [models/comments_model_dayall.joblib](models/comments_model_dayall.joblib)

Notes and tips
- Keep [models/](models/) synchronized if you retrain models locally.
- Instaloader scraping is sensitive to scraping policies and rate limits; use responsibly.
- For improved NLP features, consider switching to sentence transformers (e.g., SBERT or MiniLM).
- Model performance may vary by dataset; consider cross-validation and hyperparameter tuning for robust results.

Contributing
Contributions and ideas are welcome. Please:
1. Fork this repository.
2. Create a branch for your feature.
3. Submit a pull request describing the change.

Future improvements
- Full 60-day growth visualization and export as CSV/PDF
- SBERT or MiniLM embeddings for improved NLP features
- Docker and Nginx deployment for production readiness
- User dashboard and multi-admin support
- Support for carousel posts and Instagram Graph API integration

License
MIT License — feel free to use and modify. Add any licensing details you prefer.

Contact
Author: Praneeth

If you’d like help integrating Insta-Forecast with production environments or expanding the feature set, reach out via your preferred contact method and share any specific requirements.
