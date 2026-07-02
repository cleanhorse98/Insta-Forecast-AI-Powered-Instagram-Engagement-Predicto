# Complete Explanation: How Future Likes, Comments, and Views Are Calculated

## 📊 **How Predictions Are Made**

### **1. Overview of the Prediction Process**

The system uses **XGBoost (Gradient Boosting Decision Trees)** machine learning models to predict future engagement metrics. Here's the step-by-step process:

---

## **🔮 PREDICTION CALCULATION FLOW**

### **Step 1: Fetch Current Post Data** (`instagram_scraper.py` → `app.py`)

When you provide an Instagram URL:
1. The system extracts the **shortcode** from the URL (e.g., `ABCD1234` from `/p/ABCD1234/`)
2. Uses **Instaloader** to fetch current metadata:
   - Current likes, comments, views
   - Post timestamp (when it was published)
   - Caption text and hashtags
   - Whether it's a video or photo
   - Current post age (hours/days since posting)

### **Step 2: Build Features** (`app.py` → `build_feature_df()`)

The system creates a feature vector with **hundreds of features**:

#### **A. Time-Based Features:**
- `post_age_hours`: Hours since post was published
- `post_age_days`: Days since post was published
- `day_offset`: Rounded day number (1-60)
- `day_offset_squared`, `day_offset_log`, `day_offset_sqrt`: Time transformations

#### **B. Growth Rate Features:**
- `likes_per_day`: Current likes ÷ post age in days
- `likes_per_hour`: Current likes ÷ post age in hours
- `comments_per_day`: Current comments ÷ post age in days
- `views_per_day`: Current views ÷ post age in days
- Log transformations: `log_likes`, `log_comments`, `log_views`

#### **C. Engagement Features:**
- `engagement_rate`: (likes + comments) ÷ post age in days
- `comments_to_likes_ratio`: Comments ÷ likes
- `likes_to_views_ratio`: Likes ÷ views (for videos)
- `engagement_rate_views`: (likes + comments) ÷ views

#### **D. Content Features:**
- `is_video`: 1 if video, 0 if photo
- `seo_caption_len`: Caption character length
- `seo_word_count`: Number of words in caption
- `seo_hashtag_count`: Number of hashtags
- `seo_mention_count`: Number of @mentions
- `sentiment_score`: Positive/negative sentiment (-1 to +1)

#### **E. Text Embeddings (256 features):**
- Combined caption + hashtags are converted to text
- TF-IDF vectorization (15,000 max features)
- SVD (Truncated Singular Value Decomposition) reduces to 256 dimensions
- Creates features: `embed_0`, `embed_1`, ... `embed_255`

#### **F. Interaction Features:**
- `likes_time_interaction`: Current likes × day_offset
- `likes_log_time_interaction`: log(likes) × log(day_offset)
- Similar for comments and views

### **Step 3: Model Prediction** (`app.py` → `predict_with_model()`)

Three separate models are used:
- **Likes Model**: Predicts future likes
- **Views Model**: Predicts future views  
- **Comments Model**: Predicts future comments

Each model:
1. Takes the feature vector
2. Passes it through the trained XGBoost model
3. Outputs a raw prediction value

### **Step 4: Apply Growth Constraints** (`app.py` → `predict_with_model()`)

**Critical constraints ensure realistic predictions:**

#### **Constraint 1: Non-Negative Values**
- Predictions cannot be negative: `pred = max(pred, 0.0)`

#### **Constraint 2: Minimum Growth Rate**
Based on post age, different minimum growth rates are applied:

**For posts < 1 day old:**
- Likes: 5% per day minimum
- Views: 8% per day minimum
- Comments: 3% per day minimum

**For posts < 7 days old:**
- Likes: 2% per day minimum
- Views: 4% per day minimum
- Comments: 1% per day minimum

**For posts ≥ 7 days old:**
- Likes: 1% per day minimum
- Views: 2% per day minimum
- Comments: 0.5% per day minimum

#### **Constraint 3: Compound Growth Calculation**
For future predictions, the system calculates:
```python
days_elapsed = prediction_age_days - post_age_days
min_expected_compound = current_value * (1 + growth_rate)^days_elapsed
min_expected_linear = current_value * (1 + growth_rate * days_elapsed)
pred = max(pred, max(min_expected_compound, min_expected_linear))
```

#### **Constraint 4: Prediction Must Be ≥ Current Value**
- Future predictions **cannot be less than current values**
- `pred = max(pred, current_value * 1.0)`

#### **Constraint 5: Additional Safety Checks**
- Very new posts (< 1 day): At least 5% growth minimum
- Posts < 7 days: At least 2% growth minimum

### **Step 5: Final Formatting**

- Round predictions to integers
- Format with K/M notation (e.g., "1.5K", "2.3M")
- Calculate confidence scores based on prediction reasonableness

---

## **📈 Growth Curve Prediction** (`/predict_curve` endpoint)

For 1-60 day predictions:
1. Iterates through days 1 to 60
2. For each day:
   - Builds feature vector with that day's age
   - Predicts using the model
   - **Enforces monotonic growth** (each day ≥ previous day)
   - Applies compound growth: `pred = max(pred, prev_value * (1 + daily_growth_rate))`
3. Returns arrays of predictions for each day

---

## **🎯 Key Algorithms**

### **Machine Learning Algorithm:**
- **XGBoost** (Extreme Gradient Boosting)
- Ensemble of decision trees
- Trained on historical Instagram post data
- R² scores: ~0.998-0.999 (very high accuracy)

### **Growth Modeling:**
- Uses **compound growth** formula: `future = current × (1 + rate)^days`
- Also uses **linear growth** as fallback: `future = current × (1 + rate × days)`
- Takes the **maximum** of both to ensure realistic growth

### **Why Growth Constraints?**
Instagram engagement typically:
- Grows fastest in the first 24-48 hours
- Slows down after the first week
- Rarely decreases (once a like/view/comment exists, it doesn't disappear)
- The constraints prevent unrealistic predictions (like negative growth or sudden drops)

---

# 📁 **FILE-BY-FILE EXPLANATION**

## **1. `app.py` - Main API Backend**

**Purpose:** FastAPI web server that serves predictions via REST API

**Key Functions:**
- `fetch_post_from_instaloader()`: Fetches current post metadata from Instagram
- `build_feature_df()`: Creates feature vector from post metadata
- `predict_with_model()`: Runs prediction with growth constraints
- `predict()`: Main API endpoint for single prediction
- `predict_curve()`: API endpoint for 1-60 day growth curve
- `compute_seo()`, `clean_caption()`, `simple_sentiment()`, `compute_embeddings()`: Feature extraction utilities

**What it does:**
- Loads 3 trained models (likes, views, comments)
- Receives Instagram URL and prediction datetime
- Extracts post data, builds features, makes predictions
- Applies constraints to ensure realistic growth
- Returns formatted predictions with confidence scores

---

## **2. `train_likes.py` - Likes Model Training**

**Purpose:** Trains the XGBoost model to predict future likes

**Key Functions:**
- `validate_data_for_growth()`: Ensures training data shows correct growth patterns (no decreasing values)
- `build_features()`: Creates feature vectors from training data
- `train_xgboost_compatible()`: Trains XGBoost model with hyperparameters optimized for growth prediction
- `main()`: Orchestrates loading CSV, validation, feature building, training, and model saving

**What it does:**
- Reads CSV file with historical Instagram post data
- Validates that likes only increase over time (per post)
- Extracts features (time, engagement, text embeddings, etc.)
- Trains XGBoost model with regularization to prevent overfitting
- Saves trained model to `models/likes_model_dayall.joblib`

**Training Hyperparameters:**
- Learning rate: 0.05
- Max depth: 6
- Number of trees: 400
- Early stopping after 30 rounds without improvement

---

## **3. `train_comments.py` - Comments Model Training**

**Purpose:** Trains the XGBoost model to predict future comments

**Structure:** Identical to `train_likes.py` but:
- Target variable: `comments_count` instead of `likes`
- Features include comments-specific metrics (comments_per_day, etc.)
- Saves model to `models/comments_model_dayall.joblib`

**What it does:**
- Same training pipeline as likes model
- Predicts future comment counts based on current engagement patterns
- Uses comments-specific features and growth rates

---

## **4. `train_views.py` - Views Model Training**

**Purpose:** Trains the XGBoost model to predict future video views

**Structure:** Similar to other training scripts but:
- Target variable: `video_view_count`
- Features include views-specific metrics (views_per_day, likes_to_views_ratio)
- Only applies to video posts
- Saves model to `models/views_model_dayall.joblib`

**What it does:**
- Trains model specifically for video view prediction
- Handles video-specific engagement patterns
- Uses video-specific features (view counts, view-to-engagement ratios)

---

## **5. `instagram_scraper.py` - Instagram Data Scraper**

**Purpose:** Extracts metadata from Instagram posts using Instaloader library

**Key Functions:**
- `shortcode_from_url()`: Extracts post shortcode from Instagram URL
- `get_instagram_post_data()`: Main scraper using Instaloader
- `get_full_instagram_post_data()`: Enhanced scraper with Graph API support
- `compute_seo_features()`: Calculates SEO score and features
- `fetch_graph_api_insights()`: Fetches Instagram Graph API insights (if available)

**What it does:**
- Uses Instaloader to scrape public Instagram post data
- Extracts: caption, hashtags, likes, comments, views, timestamp
- Handles rate limiting, retries, proxy rotation
- Can log in for private posts or higher rate limits
- Optionally downloads media files
- Supports batch processing from URL files
- Exports data to Excel format

**Features:**
- Rate limit handling with exponential backoff
- Proxy pool support
- User-agent rotation
- Session file management
- Checkpoint/resume capability
- Skip existing posts (deduplication)

---

## **6. `process.py` - Data Preprocessing**

**Purpose:** Processes raw Instagram data and creates features for model training

**Key Functions:**
- `safe_parse_dt()`: Smart datetime parsing (handles various date formats)
- `normalize_series_to_utc_naive()`: Converts timezones to UTC
- `clean_caption()`: Removes URLs, converts hashtags/mentions to tokens
- `extract_hashtags()`: Parses hashtag strings into lists
- `simple_sentiment()`: Calculates positive/negative sentiment score

**What it does:**
- Reads Excel file with raw Instagram data (`output_fixed.xlsx`)
- Cleans and preprocesses captions
- Extracts hashtags and calculates sentiment
- Creates text embeddings using TF-IDF + SVD (256 dimensions)
- Calculates post age (hours since publication)
- Normalizes video view counts (0 for photos, handles missing values)
- Creates final CSV with all features (`processed1.csv`)

**Output Columns:**
- Basic: shortcode, is_video, likes, comments_count, video_view_count
- SEO: caption_len, word_count, hashtag_count, mention_count
- Time: post_age_hours
- Sentiment: sentiment_score
- Embeddings: embed_0 through embed_255 (256 features)

---

## **7. `index.html` - Frontend Web Interface**

**Purpose:** Simple web UI for making predictions

**What it does:**
- Provides input form for Instagram URL and prediction datetime
- Sends POST request to `/predict` API endpoint
- Displays predictions in formatted K/M notation
- Shows current vs. predicted values
- Displays confidence scores
- Shows hashtag analysis and recommendations
- Displays key features used in prediction

**Features:**
- Clean, minimal UI design
- Number formatting (1.5K, 2.3M format)
- Error handling for API failures
- Responsive design

---

## **8. Model Files (`models/*.joblib`)**

**Purpose:** Serialized trained machine learning models

**Files:**
- `likes_model_dayall.joblib`: Trained XGBoost model for likes prediction
- `views_model_dayall.joblib`: Trained XGBoost model for views prediction
- `comments_model_dayall.joblib`: Trained XGBoost model for comments prediction

**What they contain:**
- XGBoost model object (trained decision trees)
- List of feature names (to ensure correct feature order)
- Model metadata (training date, accuracy metrics)

**How they're used:**
- Loaded at startup in `app.py`
- Used to make predictions on new Instagram posts
- Feature vectors must match the exact feature order used during training

---

## **9. Data Files**

### **`cleaned.csv`**
- Raw/cleaned Instagram post data
- Used as input for training scripts

### **`processed.csv` / `processed1.csv`**
- Processed data with all features extracted
- Used for model training
- Contains feature vectors ready for machine learning

### **`output_fixed.xlsx`**
- Excel file with Instagram post metadata
- Output from `instagram_scraper.py`
- Input for `process.py` preprocessing

---

## **10. `README.md` - Project Documentation**

**Purpose:** Project overview and usage instructions

**Contains:**
- Project description
- File structure explanation
- Quick start guide
- API endpoint documentation
- Training instructions
- Model performance metrics
- Future improvement ideas

---

## **11. `requirements.txt` - Python Dependencies**

**Purpose:** Lists all required Python packages

**Key dependencies:**
- `fastapi`: Web framework for API
- `xgboost`: Machine learning library
- `pandas`, `numpy`: Data processing
- `instaloader`: Instagram scraping
- `scikit-learn`: Feature engineering (TF-IDF, SVD)
- `joblib`: Model serialization
- `uvicorn`: ASGI server

---

## **12. `FEATURE_SUGGESTIONS.md`**

**Purpose:** Documentation of potential new features to improve predictions

**Contains:**
- Ideas for additional features
- Feature engineering suggestions
- Model improvement recommendations

---

# **🔄 Complete Workflow**

## **Training Pipeline:**
1. **Scrape Data** (`instagram_scraper.py`) → Collect Instagram post metadata
2. **Process Data** (`process.py`) → Extract features, create embeddings
3. **Train Models** (`train_*.py`) → Train XGBoost models on processed data
4. **Save Models** → Store models in `models/` directory

## **Prediction Pipeline:**
1. **User Input** → Provides Instagram URL + prediction datetime
2. **Fetch Current Data** → Scrape current post metadata
3. **Build Features** → Create feature vector matching training data
4. **Predict** → Use trained models to get raw predictions
5. **Apply Constraints** → Ensure realistic growth patterns
6. **Format & Return** → Return formatted predictions with confidence scores

---

# **🎓 Key Concepts**

### **Why XGBoost?**
- Handles complex non-linear relationships
- Works well with mixed data types (numbers, categories, text embeddings)
- Provides high accuracy (R² ~0.998)
- Fast inference time

### **Why Growth Constraints?**
- Instagram engagement only increases over time (likes/comments don't decrease)
- Posts grow fastest in first 24-48 hours
- Prevents unrealistic predictions (negative growth, sudden drops)
- Ensures predictions are always >= current values

### **Why Multiple Features?**
- **Time features**: Capture growth patterns over time
- **Growth rate features**: Measure current engagement velocity
- **Content features**: Caption length, hashtags, sentiment affect engagement
- **Text embeddings**: Capture semantic meaning of captions/hashtags
- **Interaction features**: Model relationships between metrics and time

### **Why Separate Models?**
- Likes, views, and comments have different growth patterns
- Videos have different engagement than photos
- Separate models allow specialized training for each metric
- Better accuracy than single multi-output model

---

# **📊 Example Calculation**

**Given:**
- Post URL: `https://www.instagram.com/p/ABC123/`
- Current likes: 1,000
- Post age: 2 days
- Prediction time: 7 days from now (5 days in the future)

**Process:**
1. Fetch current data: 1,000 likes, 2 days old
2. Build features: age=2 days, likes_per_day=500, etc.
3. Model predicts: 3,500 likes (raw prediction)
4. Apply constraints:
   - Minimum growth: 1,000 × (1.02)^5 = 1,104 (for 2% daily growth)
   - Max with current: max(3,500, 1,000) = 3,500
   - Final prediction: 3,500 likes (or higher if constraints require)

**Result:**
- Current: 1,000 likes
- Predicted (7 days): 3,500 likes
- Growth: 250% increase over 5 days

---

This system provides accurate, realistic predictions by combining machine learning with domain knowledge about Instagram engagement patterns!

