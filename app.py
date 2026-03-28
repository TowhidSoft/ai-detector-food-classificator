from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import numpy as np
import open_clip
from open_clip import tokenize
import tempfile
import os
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ----------------------------
# Global variables for model
# ----------------------------
model = None
preprocess = None
detection_features = None
meal_features = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Detection Prompts
# ----------------------------
FOOD_DETECTION_PROMPTS = [
    "a plate of food",
    "eating food",
    "delicious meal",
    "food on table",
    "chef cooking food"
]

NOT_FOOD_PROMPTS = [
    "a beautiful landscape",
    "people talking",
    "car driving",
    "person running",
    "office work"
]

ALL_DETECTION_PROMPTS = FOOD_DETECTION_PROMPTS + NOT_FOOD_PROMPTS

def get_model():
    global model, preprocess, detection_features, meal_features
    
    if model is not None:
        return model, preprocess, detection_features, meal_features

    logger.info("Loading model...")
    global device
    # Create model
    m, _, p = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = m.to(device).eval()
    preprocess = p
    
    # Precompute Prompt Features
    logger.info("Computing prompt features...")
    
    # 1. Detection Features
    with torch.no_grad():
        detection_tokens = tokenize(ALL_DETECTION_PROMPTS).to(device)
        df = model.encode_text(detection_tokens)
        df /= df.norm(dim=-1, keepdim=True)
        detection_features = df

    # 2. Meal Features
    with torch.no_grad():
        meal_tokens = tokenize(MEAL_PROMPTS_FLAT).to(device)
        mf = model.encode_text(meal_tokens)
        mf /= mf.norm(dim=-1, keepdim=True)
        meal_features = mf
        
    logger.info("Model loaded successfully.")
    return model, preprocess, detection_features, meal_features

# ----------------------------
# Categories for CLASSIFICATION (Time-based targeting)
# ----------------------------
MEAL_CATEGORIES_MAP = {
    "breakfast_meal": [
        "breakfast food like eggs pancakes or cereal",
        "drinking coffee in the morning",
        "morning toast and tea"
    ],
    "lunch_meal": [
        "lunch meal like a sandwich or burger",
        "eating rice and curry for lunch",
        "a hearty lunch bowl"
    ],
    "dinner_meal": [
        "dinner meal like steak or pasta",
        "family having dinner together",
        "evening supper feast"
    ]
}

# Flatten for embedding
MEAL_KEYS = list(MEAL_CATEGORIES_MAP.keys()) 
MEAL_PROMPTS_FLAT = []
MEAL_KEY_INDICES = [] # To map back which prompt belongs to which key

for key in MEAL_KEYS:
    for prompt in MEAL_CATEGORIES_MAP[key]:
        MEAL_PROMPTS_FLAT.append(prompt)
        MEAL_KEY_INDICES.append(key)

# ----------------------------
# Video analysis function
# ----------------------------
def analyze_video(video_path):
    # Ensure model is loaded
    model, preprocess, detection_features, meal_features = get_model()
    
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Analyzing video: {duration:.2f}s, {total_frames} frames, {fps} fps")

    # Default frame interval
    frame_interval = 30
    
    # Adjust frame interval for long videos to avoid Cloudflare/Render timeouts
    if duration > 120:  # > 2 mins
        frame_interval = max(int(fps * 2), 60) # 1 frame every 2 seconds
    elif duration > 30: # > 30s
        frame_interval = max(int(fps), 30)    # 1 frame per second
    
    logger.info(f"Using frame_interval: {frame_interval}")

    food_frame_count = 0
    total_analyzed_frames = 0
    food_frame_features = []
    
    current_frame = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        total_analyzed_frames += 1
        if total_analyzed_frames % 5 == 0:
             logger.info(f"Analyzed {total_analyzed_frames} frames (at {current_frame}/{total_frames})...")

        # Convert frame to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # ---------------------------
            # 1. Per-Frame Detection
            # ---------------------------
            similarity = (100.0 * image_features @ detection_features.T).softmax(dim=-1)
            probs = similarity.cpu().numpy()[0]
            
            # Sum prob of all 'Food' prompts vs all 'Not Food' prompts
            n_food = len(FOOD_DETECTION_PROMPTS)
            prob_food = np.sum(probs[:n_food])
            prob_not_food = np.sum(probs[n_food:])
            
            if prob_food > prob_not_food:
                food_frame_count += 1
                food_frame_features.append(image_features)
        
        current_frame += frame_interval

    cap.release()
    
    if total_analyzed_frames == 0:
        return {"category": "not_food_content", "is_food": False}

    # ---------------------------
    # Global Decision
    # ---------------------------
    food_ratio = food_frame_count / total_analyzed_frames
    is_food = food_ratio > 0.15 
    
    if not is_food:
         return {
            "category": "not_food_content", 
            "is_food": False, 
            "confidence": round(float(food_ratio), 3),
            "debug_info": f"Only {food_frame_count}/{total_analyzed_frames} frames detected as food."
        }

    # ---------------------------
    # Meal Classification (on food frames only)
    # ---------------------------
    if not food_frame_features:
        best_category = "lunch_meal" # Fallback
    else:
        # Average features of CONFIRMED food frames only
        avg_food_features = torch.cat(food_frame_features, dim=0).mean(dim=0, keepdim=True)
        avg_food_features /= avg_food_features.norm(dim=-1, keepdim=True)
        
        meal_similarity = (100.0 * avg_food_features @ meal_features.T).softmax(dim=-1)
        meal_probs = meal_similarity.cpu().numpy()[0]
        
        category_scores = {k: 0.0 for k in MEAL_KEYS}
        for idx, score in enumerate(meal_probs):
            cat_key = MEAL_KEY_INDICES[idx]
            category_scores[cat_key] += score
            
        best_category = max(category_scores, key=category_scores.get)

    # Time logic mapping
    time_mapping = {
        "breakfast_meal": ["morning"],
        "lunch_meal": ["afternoon"],
        "dinner_meal": ["evening", "night"]
    }

    return {
        "category": best_category,
        "is_food": True,
        "sub_category": best_category,
        "food_score": round(float(food_ratio), 3),
        "recommended_times": time_mapping.get(best_category, []),
        "debug_info": f"Classified based on {food_frame_count}/{total_analyzed_frames} food frames."
    }

# ----------------------------
# FastAPI endpoints
# ----------------------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading model at startup...")
    get_model()
    logger.info("Startup sequence complete.")

@app.get("/health")
def health():
    return {"status": "alive"}

@app.post("/analyze-video")
async def analyze_video_api(video: UploadFile = File(...)):
    logger.info(f"Received video analysis request for file: {video.filename}")
    
    # Save uploaded video temporarily using chunks to avoid memory overflow
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            while chunk := await video.read(1024 * 1024): # 1MB chunks
                temp.write(chunk)
            temp_path = temp.name
        
        logger.info(f"Video saved to {temp_path}. Starting classification...")
        result = analyze_video(temp_path)
        logger.info("Analysis finished successfully.")
        return result
        
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}", exc_info=True)
        return {"error": "Internal server error during video analysis.", "details": str(e)}
        
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Temporary file {temp_path} removed.")
