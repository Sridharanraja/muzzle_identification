import streamlit as st
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from ultralytics import YOLO
YOLO_AVAILABLE = True
import pandas as pd
import io, base64, os, zipfile, json, shutil
try:
    from pymongo import MongoClient, ASCENDING
    PYMONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    ASCENDING = None
    PYMONGO_AVAILABLE = False
from datetime import datetime
import sys

# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------------------------------------------
# MongoDB Database Connection (REQUIRED)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db():
    """Connect to MongoDB - Required for application functionality"""
    if not PYMONGO_AVAILABLE:
        st.error("❌ pymongo is not installed. Please install it to use this application.")
        st.stop()
    
    try:
        # Check if MongoDB URI is provided in environment or secrets
        mongodb_uri = os.environ.get("MONGODB_URI")
        if not mongodb_uri and "mongodb" in st.secrets and "uri" in st.secrets["mongodb"]:
            mongodb_uri = st.secrets["mongodb"]["uri"]
        
        if not mongodb_uri:
            st.error("❌ MongoDB URI not configured. Please set MONGODB_URI environment variable.")
            st.stop()
        
        client = MongoClient(mongodb_uri, connectTimeoutMS=20000, serverSelectionTimeoutMS=20000)
        db = client["cattle_db"]
        
        # Create indexes for better performance
        if ASCENDING is not None:
            db["cattle_images"].create_index([("12_digit_id", ASCENDING)], unique=True)
            db["yolo_results"].create_index([("image_id", ASCENDING)], unique=True)
        
        # Test connection
        client.admin.command('ping')
        return db
    except Exception as e:
        st.error(f"❌ Failed to connect to MongoDB: {e}")
        st.stop()

# Connect to MongoDB (required)
db = get_db()
cattle_collection = db["cattle_images"]
yolo_collection = db["yolo_results"]
faiss_index_collection = db["faiss_index"]

# -----------------------------------------------------------------------------
# Load Data.csv
# -----------------------------------------------------------------------------
@st.cache_data
def load_csv():
    if os.path.exists("./file/data.csv"):
        df = pd.read_csv("./file/data.csv", dtype=str)
        # Clean up the 12_digit_id column
        df["12_digit_id"] = (
            df["12_digit_id"]
            .str.replace(r"\.0$", "", regex=True)   # remove trailing .0 if present
            .str.replace(",", "", regex=True)       # remove commas
        )
        
        def fix_id(x):
            try:
                return str(int(float(x)))
            except:
                return x
        
        df["12_digit_id"] = df["12_digit_id"].apply(fix_id)
        return df
    else:
        return pd.DataFrame()

cattle_df = load_csv()

# -----------------------------------------------------------------------------
# Load CLIP Model with error handling
# -----------------------------------------------------------------------------
@st.cache_resource
def load_clip_model():
    """Load CLIP model with caching and error handling"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use ViT-B/16 for higher resolution and better accuracy
        # model, preprocess = clip.load("ViT-B/16", device=device)
        return model, preprocess, device
    except Exception as e:
        st.error(f"Failed to load CLIP model: {str(e)}")
        st.info("This might be due to missing dependencies. Please ensure all requirements are installed.")
        st.stop()

# Initialize CLIP model
model, preprocess, device = load_clip_model()

# -----------------------------------------------------------------------------
# Load YOLO Models
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_models():
    if not YOLO_AVAILABLE:
        return None, None
    try:
        roi_model = YOLO("./models/roi_best_600.pt")   # ROI detection model new  
        cls_model = YOLO("./models/best_25_train8.pt")  # Classification model
        return roi_model, cls_model
    except Exception as e:
        st.warning(f"YOLO model loading error: {str(e)[:200]}")
        return None, None

roi_model, cls_model = load_yolo_models()

# -----------------------------------------------------------------------------
# FAISS Index Setup (MongoDB backed)
# -----------------------------------------------------------------------------
# FAISS index now stored in MongoDB (no local files needed)

# Create necessary directories
os.makedirs(".streamlit", exist_ok=True)

# Initialize FAISS index
embedding_dim = 512
index = faiss.IndexFlatIP(embedding_dim)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def embed_image(img):
    """Convert image to CLIP embedding"""
    try:
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def save_yolo_result(image_id: str, roi_conf: float, class_name: str, confidence: float, roi_bbox: list):
    """Save YOLO classification result to MongoDB"""
    try:
        doc = {
            "image_id": image_id,
            "roi_confidence": roi_conf,
            "class_name": class_name,
            "classification_confidence": confidence,
            "roi_bbox": roi_bbox,
            "timestamp": datetime.utcnow().isoformat()
        }
        yolo_collection.replace_one({"image_id": image_id}, doc, upsert=True)
        return True
    except Exception as e:
        st.error(f"Error saving YOLO result: {str(e)}")
        return False

def get_yolo_result(image_id: str):
    """Get YOLO classification result from MongoDB"""
    try:
        return yolo_collection.find_one({"image_id": image_id})
    except Exception as e:
        st.error(f"Error retrieving YOLO result: {str(e)}")
        return None

def show_images_with_captions(img_paths, title="Reference Image", from_db=False):
    """Display images with captions"""
    try:
        if from_db:
            # Handle base64 encoded images from database
            if isinstance(img_paths, list):
                for i, img_data in enumerate(img_paths):
                    if isinstance(img_data, dict) and 'b64' in img_data:
                        try:
                            raw = base64.b64decode(img_data['b64'])
                            if len(raw) > 0:
                                st.image(Image.open(io.BytesIO(raw)), caption=img_data.get('filename', f'Ref {i+1}'), width=120)
                            else:
                                st.error(f"❌ Empty image data: {img_data.get('filename', f'Ref {i+1}')}")
                        except Exception as e:
                            st.error(f"❌ Invalid image: {img_data.get('filename', f'Ref {i+1}')} - {str(e)[:50]}")
                            continue
            return
        
        # Handle file paths (existing functionality)
        valid_paths = [p for p in (img_paths if isinstance(img_paths, list) else [img_paths]) if os.path.exists(p)]
        
        if not valid_paths:
            st.warning("No valid image paths found")
            return
            
        if len(valid_paths) > 1:
            captions = [f"Ref {i+1}" for i in range(len(valid_paths))]
            st.image(valid_paths, caption=captions, width=120)
        else:
            st.image(valid_paths[0], caption=title, width=120)
    except Exception as e:
        st.error(f"Error displaying images: {str(e)}")

# -----------------------------------------------------------------------------
# MongoDB Helper Functions
# -----------------------------------------------------------------------------
def save_new_cattle_to_db(cattle_id: str, cattle_name: str, cattle_class: str, image_files, embeddings=None):
    """Save new cattle to MongoDB with images and embeddings"""
    # MongoDB is now required, no need to check
        
    created_at = datetime.utcnow().isoformat()
    image_entries = []
    
    for i, f in enumerate(image_files, start=1):
        try:
            if hasattr(f, 'read'):
                # File upload object
                f.seek(0)  # Reset file pointer FIRST
                raw = f.read()
                if len(raw) == 0:
                    st.error(f"Warning: File {getattr(f, 'name', 'unknown')} is empty")
                    continue
                ext = os.path.splitext(getattr(f, 'name', 'image'))[1] or ".jpg"
            else:
                # PIL Image object
                buf = io.BytesIO()
                f.save(buf, format='JPEG')
                raw = buf.getvalue()
                if len(raw) == 0:
                    st.error(f"Warning: Could not convert PIL image to bytes")
                    continue
                ext = ".jpg"
            
            b64 = base64.b64encode(raw).decode("utf-8")
            filename = f"{cattle_id}_{i}{ext}"
            image_entries.append({"filename": filename, "b64": b64})
        except Exception as e:
            st.error(f"Error processing image {i}: {str(e)}")
            continue

    doc = {
        "12_digit_id": cattle_id,
        "cattle_name": cattle_name,
        "cattle_class": cattle_class,
        "images": image_entries,
        "embedding": embeddings.flatten().tolist() if embeddings is not None else None,
        "created_at": created_at
    }
    
    try:
        cattle_collection.insert_one(doc)
        return doc
    except Exception as e:
        st.error(f"Error saving to MongoDB: {e}")
        return None

def get_cattle_by_id(cattle_id: str):
    """Get cattle record by ID from MongoDB"""
    return cattle_collection.find_one({"12_digit_id": cattle_id})

def list_cattle_from_db(filter_id=None, filter_name=None, limit=100):
    """List cattle from MongoDB with optional filters"""
        
    q = {}
    if filter_id:
        q["12_digit_id"] = {"$regex": f"^{filter_id}"}
    if filter_name:
        q["cattle_name"] = {"$regex": filter_name, "$options": "i"}
    
    try:
        cursor = cattle_collection.find(q).sort("created_at", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        st.error(f"Error querying MongoDB: {e}")
        return []

def update_cattle_in_db(cattle_id: str, updates: dict):
    """Update cattle record in MongoDB"""
    if cattle_collection is None:
        return False
    
    try:
        result = cattle_collection.update_one(
            {"12_digit_id": cattle_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Error updating cattle: {e}")
        return False

def delete_cattle_image_from_db(cattle_id: str, image_filename: str):
    """Remove a specific image from cattle record"""
    if cattle_collection is None:
        return False
    
    try:
        result = cattle_collection.update_one(
            {"12_digit_id": cattle_id},
            {"$pull": {"images": {"filename": image_filename}}}
        )
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Error removing image: {e}")
        return False

def add_cattle_images_to_db(cattle_id: str, new_images):
    """Add new images to existing cattle record"""
    if cattle_collection is None:
        return False
    
    try:
        # Get current cattle record
        cattle = get_cattle_by_id(cattle_id)
        if not cattle:
            return False
        
        # Get current image count
        current_images = cattle.get("images", [])
        start_idx = len(current_images) + 1
        
        # Process new images
        image_entries = []
        for i, f in enumerate(new_images, start=start_idx):
            try:
                if hasattr(f, 'read'):
                    f.seek(0)  # Reset file pointer FIRST
                    raw = f.read()
                    if len(raw) == 0:
                        st.error(f"Warning: File {getattr(f, 'name', 'unknown')} is empty")
                        continue
                    ext = os.path.splitext(getattr(f, 'name', 'image'))[1] or ".jpg"
                else:
                    buf = io.BytesIO()
                    f.save(buf, format='JPEG')
                    raw = buf.getvalue()
                    if len(raw) == 0:
                        st.error(f"Warning: Could not convert PIL image to bytes")
                        continue
                    ext = ".jpg"
                
                b64 = base64.b64encode(raw).decode("utf-8")
                filename = f"{cattle_id}_{i}{ext}"
                image_entries.append({"filename": filename, "b64": b64})
            except Exception as e:
                st.error(f"Error processing additional image {i}: {str(e)}")
                continue
        
        # Add new images to database
        result = cattle_collection.update_one(
            {"12_digit_id": cattle_id},
            {"$push": {"images": {"$each": image_entries}}}
        )
        return result.modified_count > 0
    except Exception as e:
        st.error(f"Error adding images: {e}")
        return False

def save_faiss_to_mongodb(index, ordered_ids):
    """Save FAISS index and ordered IDs to MongoDB"""
    try:
        # Serialize FAISS index to bytes
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp:
            faiss.write_index(index, tmp.name)
            tmp.flush()
            # Read back as bytes
            with open(tmp.name, 'rb') as f:
                index_bytes = f.read()
            os.unlink(tmp.name)
        
        # Store in MongoDB
        faiss_index_collection.update_one(
            {"_id": "faiss_index"},
            {
                "$set": {
                    "index_data": index_bytes,
                    "ordered_ids": ordered_ids,
                    "updated_at": datetime.now()
                }
            },
            upsert=True
        )
        return True
    except Exception as e:
        st.error(f"Error saving FAISS index to MongoDB: {e}")
        return False

def load_faiss_from_mongodb():
    """Load FAISS index and ordered IDs from MongoDB"""
    global index
    try:
        doc = faiss_index_collection.find_one({"_id": "faiss_index"})
        if doc and "index_data" in doc:
            # Write bytes to temporary file (FAISS requirement)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as tmp:
                tmp.write(doc["index_data"])
                tmp.flush()
                # Load index from file
                index = faiss.read_index(tmp.name)
                os.unlink(tmp.name)
            
            ordered_ids = doc.get("ordered_ids", [])
            return index, ordered_ids
        return None, []
    except Exception as e:
        st.error(f"Error loading FAISS index from MongoDB: {e}")
        return None, []

def get_all_cattle_embeddings():
    """Get all cattle embeddings for FAISS index in sorted order"""
    try:
        # Always sort by 12_digit_id for consistent ordering
        cursor = cattle_collection.find(
            {"embedding": {"$ne": None}}, 
            {"12_digit_id": 1, "embedding": 1}
        ).sort("12_digit_id", 1)  # Sort ascending by ID
        
        # Return both ordered list and dict
        ordered_ids = []
        embeddings_dict = {}
        for doc in cursor:
            cattle_id = doc["12_digit_id"]
            ordered_ids.append(cattle_id)
            embeddings_dict[cattle_id] = doc["embedding"]
        
        return ordered_ids, embeddings_dict
    except Exception as e:
        st.error(f"Error fetching embeddings: {e}")
        return [], {}

def rebuild_faiss():
    """Rebuild FAISS index from MongoDB embeddings with consistent ordering"""
    global index
    try:
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Get embeddings from MongoDB in sorted order
        ordered_ids, embeddings_dict = get_all_cattle_embeddings()
        
        # Build index in the same order as IDs
        for cattle_id in ordered_ids:
            embedding = embeddings_dict[cattle_id]
            if embedding:
                emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                index.add(emb)
        
        # Save FAISS index to MongoDB
        save_faiss_to_mongodb(index, ordered_ids)
                    
        return True
    except Exception as e:
        st.error(f"Error rebuilding FAISS index: {str(e)}")
        return False

def create_metadata_csv_bytes(docs):
    rows = []
    for d in docs:
        cid = d.get("12_digit_id")
        name = d.get("cattle_name")
        created = d.get("created_at")
        for img in d.get("images", []):
            rows.append({
                "12_digit_id": cid,
                "cattle_name": name,
                "created_at": created,
                "image_filename": img.get("filename")
            })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()

def create_images_zip_bytes(docs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for d in docs:
            for img in d.get("images", []):
                filename = img.get("filename")
                b64 = img.get("b64")
                raw = base64.b64decode(b64)
                z.writestr(filename, raw)
    buf.seek(0)
    return buf.read()

# Load existing FAISS index if available
# Load FAISS index from MongoDB or create new one
try:
    loaded_index, loaded_ids = load_faiss_from_mongodb()
    if loaded_index is not None:
        index = loaded_index
    else:
        index = faiss.IndexFlatIP(embedding_dim)
        rebuild_faiss()
except Exception as e:
    st.warning(f"Could not load FAISS index from MongoDB: {str(e)}. Will rebuild.")
    index = faiss.IndexFlatIP(embedding_dim)
    rebuild_faiss()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🐄 Livestock Management System")
st.caption("Comprehensive cattle identification using CLIP embeddings, YOLO detection, and MongoDB storage")

# Display system information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Device", device.upper())
with col2:
    try:
        db_count = cattle_collection.count_documents({})
        st.metric("MongoDB Records", db_count)
    except:
        st.metric("MongoDB Records", "Error")
with col3:
    st.metric("CSV Entries", len(cattle_df) if not cattle_df.empty else 0)

# Create tabs for different functionality
tabs = st.tabs(["🔍 CLIP Identification", "📊 YOLO Classification", "➕ Register Cattle", "📂 Browse & Download", "🔎 Quick Lookup", "🛠️ Management", "🗄️ Database Viewer"])

# ================== TAB: CLIP Identification ==================
with tabs[0]:
    st.header("CLIP-based Cattle Identification")
    
    # Check if we have any registered cattle
    total_cattle = 0
    try:
        total_cattle = cattle_collection.count_documents({})
    except:
        total_cattle = 0
    
    if total_cattle == 0:
        st.info("📝 No cattle registered yet. Please register cattle first.")
    else:
        # Search Parameters (show before image upload)
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            k = st.slider("Top-K results", 1, max(2, min(5, total_cattle)), 1, key="clip_k")
        with col_param2:
            threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.75, step=0.05, key="clip_threshold")

        test_file = st.file_uploader(
            "Upload New Image to Identify", 
            type=["jpg","jpeg","png"], 
            key="test",
            help="Upload a clear image of cattle muzzle for identification"
        )

        if test_file:
            try:
                test_img = Image.open(test_file).convert("RGB")
                st.image(test_img, caption="Test Image", width=300)

                if st.button("🔍 Identify Cattle", type="primary"):
                    with st.spinner("Processing image and searching..."):
                        test_feat = embed_image(test_img)
                        
                        if test_feat is not None and index.ntotal > 0:
                            # Normalize test feature
                            test_feat = test_feat / np.linalg.norm(test_feat, axis=1, keepdims=True)
                            
                            # Search in FAISS index
                            D, I = index.search(test_feat, k)
                            
                            st.subheader("🎯 Identification Results")
                            
                            # Get IDs from MongoDB in same order as FAISS index
                            try:
                                # Load the ordered IDs from MongoDB
                                faiss_doc = faiss_index_collection.find_one({"_id": "faiss_index"})
                                if faiss_doc and "ordered_ids" in faiss_doc:
                                    ids = faiss_doc["ordered_ids"]
                                    
                                    # Get details for all IDs
                                    docs = list(cattle_collection.find(
                                        {"12_digit_id": {"$in": ids}},
                                        {"12_digit_id": 1, "cattle_name": 1, "cattle_class": 1, "images": 1}
                                    ))
                                    details_map = {doc["12_digit_id"]: doc for doc in docs}
                                else:
                                    # Fallback: rebuild index if not found
                                    rebuild_faiss()
                                    faiss_doc = faiss_index_collection.find_one({"_id": "faiss_index"})
                                    if faiss_doc and "ordered_ids" in faiss_doc:
                                        ids = faiss_doc["ordered_ids"]
                                        docs = list(cattle_collection.find(
                                            {"12_digit_id": {"$in": ids}},
                                            {"12_digit_id": 1, "cattle_name": 1, "cattle_class": 1, "images": 1}
                                        ))
                                        details_map = {doc["12_digit_id"]: doc for doc in docs}
                                    else:
                                        ids = []
                                        details_map = {}
                            except Exception as e:
                                st.error(f"Error loading cattle IDs: {e}")
                                ids = []
                                details_map = {}
                            
                            found_match = False
                            
                            for rank, (score, idx) in enumerate(zip(D[0], I[0])):
                                if idx == -1 or idx >= len(ids):
                                    continue
                                    
                                cattle_id = ids[idx]
                                details = details_map[cattle_id]
                                
                                if score >= threshold:
                                    found_match = True
                                    with st.container():
                                        class_info = details.get('cattle_class', details.get('class', 'Unknown'))
                                        name_info = details.get('cattle_name', details.get('name', 'Unknown'))
                                        st.success(
                                            f"✅ **Match {rank+1}** (Confidence: {score:.3f})\n\n"
                                            f"🆔 **ID:** {cattle_id}\n\n"
                                            f"🐂 **Class:** {class_info}\n\n"
                                            f"📛 **Name:** {name_info}"
                                        )
                                        if 'images' in details:
                                            show_images_with_captions(details["images"], title="Reference Images", from_db=True)
                                        st.divider()
                                else:
                                    name_info = details.get('cattle_name', details.get('name', 'Unknown'))
                                    st.warning(f"❌ Candidate {rank+1}: {name_info} (Low confidence: {score:.3f})")
                            
                            if not found_match:
                                st.error(f"❌ No confident matches found above threshold {threshold:.2f}")
                                st.info("💡 Try lowering the confidence threshold or register this cattle if it's new.")
                        else:
                            st.error("❌ Failed to process image or empty database")
                            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")

# ================== TAB: YOLO Classification ==================
with tabs[1]:
    st.header("YOLO-based Cattle Classification")
    
    if roi_model is None or cls_model is None:
        st.error("⚠️ YOLO models are incompatible with standard ultralytics versions. They require a custom version that is not available in this deployment environment.")
    else:
        ui_threshold = st.slider("Classification Confidence Threshold", 0.0, 1.0, 0.5, 0.01, key="yolo_threshold")
        uploaded_file = st.file_uploader("Upload an Image for YOLO Classification", type=["jpg", "jpeg", "png"], key="yolo_upload")

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Step 1: ROI detection
            roi_results = roi_model.predict(image)
            if not roi_results or len(roi_results[0].boxes) == 0:
                st.warning("⚠️ Please upload a proper cow face image (no ROI detected).")
            else:
                # Get the highest confidence ROI
                roi_box = max(roi_results[0].boxes, key=lambda b: b.conf)
                roi_conf = float(roi_box.conf)

                if roi_conf < 0.60:
                    st.warning("⚠️ Please upload a proper cow face image (ROI confidence < 0.60).")
                else:
                    # Crop ROI region
                    x1, y1, x2, y2 = map(int, roi_box.xyxy[0].tolist())
                    roi_crop = image.crop((x1, y1, x2, y2))
                    st.image(roi_crop, caption=f"Detected ROI (Confidence: {roi_conf:.2f})", use_column_width=True)

                    # Step 2: Classification on ROI
                    results = cls_model.predict(roi_crop)
                    top_result = results[0].probs
                    class_id = int(top_result.top1)
                    confidence = float(top_result.top1conf)
                    class_name = cls_model.names[class_id]

                    if confidence < 0.90:
                        st.error("⚠️ Data not available in DB for reliable classification.")
                    else:
                        # Store YOLO results in MongoDB
                        import hashlib
                        # Read file content BEFORE it's consumed by PIL
                        uploaded_file.seek(0)  # Reset to beginning first
                        file_content = uploaded_file.read()
                        image_id = hashlib.md5(file_content).hexdigest()
                        uploaded_file.seek(0)  # Reset for later use
                        
                        # Save YOLO result to MongoDB
                        # save_yolo_result(
                        #     image_id=image_id,
                        #     roi_conf=roi_conf,
                        #     class_name=class_name,
                        #     confidence=confidence,
                        #     roi_bbox=[x1, y1, x2, y2]
                        # )
                        
                        # st.success(f"✅ Classification Result Saved to MongoDB")
                        # st.info(f"Class: {class_name} | Confidence: {confidence:.2%}")
                        
                        # Also show classification results
                        if confidence >= ui_threshold:
                            st.success(f"Predicted Class: **{class_name}** (Confidence: {confidence:.2f})")
                            if not cattle_df.empty:
                                row = cattle_df[cattle_df["class"] == class_name]
                                if not row.empty:
                                    st.write("📌 **Mapped Result from CSV:**")
                                    st.table(row[["12_digit_id", "cattle_name", "class"]])
                                else:
                                    st.warning("⚠️ Predicted class not found in CSV mapping.")
                            else:
                                st.info("CSV data not available for mapping.")
                        else:
                            st.warning("No prediction passed the selected confidence threshold.")

# ================== TAB: Registration ==================
with tabs[2]:
    st.header("Register New Cattle")

    with st.form("registration_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            cattle_id = st.text_input("Enter 12-digit Cattle Code", help="Unique identifier for the cattle")
        with col2:
            cattle_class = st.text_input("Enter Cattle Class", help="e.g., Holstein, Angus, etc.")
        with col3:
            cattle_name = st.text_input("Enter Cattle Name", help="Name or identifier for the cattle")

        ref_files = st.file_uploader(
            "Upload Reference Images", 
            type=["jpg","jpeg","png"], 
            accept_multiple_files=True,
            help="Upload 2-5 clear images of the cattle's muzzle"
        )
        
        submitted = st.form_submit_button("Register Cattle")

    if submitted:
        if not (cattle_id and cattle_class and cattle_name and ref_files):
            st.error("⚠️ Please enter all details and upload at least one image.")
        elif len(cattle_id) != 12 or not cattle_id.isdigit():
            st.error("⚠️ Cattle code must be exactly 12 digits.")
        elif get_cattle_by_id(cattle_id):
            st.warning("⚠️ This 12-digit code is already registered!")
        else:
            with st.spinner("Processing images and validating cattle ROI..."):
                try:
                    embeddings = []
                    img_paths = []
                    # No longer need local file storage
                    # All images are stored in MongoDB
                    
                    # Track valid and invalid images with image data
                    valid_images = []
                    invalid_images = []
                    all_validation_results = []

                    # Process each uploaded image with ROI validation
                    for i, file in enumerate(ref_files):
                        img = Image.open(file).convert("RGB")
                        
                        # Apply ROI validation if YOLO model is available
                        has_valid_roi = False
                        roi_confidence = 0.0
                        validation_status = ""
                        
                        if roi_model is not None:
                            roi_results = roi_model.predict(img)
                            if roi_results and len(roi_results[0].boxes) > 0:
                                # Get the highest confidence ROI
                                roi_box = max(roi_results[0].boxes, key=lambda b: b.conf)
                                roi_confidence = float(roi_box.conf)
                                
                                if roi_confidence >= 0.60:
                                    has_valid_roi = True
                                    validation_status = f"✅ Valid (ROI: {roi_confidence:.2f})"
                                    valid_images.append((file.name, roi_confidence, img))
                                else:
                                    validation_status = f"❌ Invalid (ROI: {roi_confidence:.2f} < 0.60)"
                                    invalid_images.append((file.name, f"Low ROI confidence: {roi_confidence:.2f}", img))
                            else:
                                validation_status = "❌ Invalid (No ROI detected)"
                                invalid_images.append((file.name, "No cattle ROI detected", img))
                        else:
                            # If YOLO is not available, accept all images
                            has_valid_roi = True
                            validation_status = "✅ Valid (YOLO not available)"
                            valid_images.append((file.name, "YOLO not available - accepted", img))
                        
                        # Store validation result for display
                        all_validation_results.append({
                            'image': img,
                            'name': file.name,
                            'status': validation_status,
                            'valid': has_valid_roi,
                            'confidence': roi_confidence if roi_model is not None else None
                        })
                        
                        # Only process embeddings for images with valid ROI
                        if has_valid_roi:
                            emb = embed_image(img)
                            
                            if emb is not None:
                                embeddings.append(emb)
                                # Images will be saved to MongoDB
                                img_paths.append(img)  # Store PIL image directly
                    
                    # Display validation results with images
                    st.subheader("Image Validation Results")
                    
                    # Display valid images
                    if valid_images:
                        st.success(f"✅ {len(valid_images)} Valid Image(s)")
                        valid_cols = st.columns(min(len(valid_images), 3))
                        for idx, (name, conf, img) in enumerate(valid_images):
                            with valid_cols[idx % 3]:
                                st.image(img, caption=f"✅ {name}", use_container_width=True)
                                if isinstance(conf, float):
                                    st.caption(f"ROI Confidence: {conf:.2f}")
                                else:
                                    st.caption(str(conf))
                    
                    # Display invalid images
                    if invalid_images:
                        st.error(f"❌ {len(invalid_images)} Invalid Image(s) - These will NOT be saved")
                        st.warning("⚠️ Invalid images will not be uploaded to the database. Please upload different images with clear cattle muzzle visible.")
                        invalid_cols = st.columns(min(len(invalid_images), 3))
                        for idx, (name, reason, img) in enumerate(invalid_images):
                            with invalid_cols[idx % 3]:
                                st.image(img, caption=f"❌ {name}", use_container_width=True)
                                st.caption(reason)
                    
                    # Summary
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Valid Images", len(valid_images), delta=None, delta_color="normal")
                    with col2:
                        st.metric("Invalid Images", len(invalid_images), delta=None, delta_color="normal")

                    if embeddings:
                        # Calculate average embedding
                        avg_embedding = np.mean(np.vstack(embeddings), axis=0, keepdims=True)
                        # Normalize the average embedding
                        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding, axis=1, keepdims=True)

                        # Save to MongoDB (only valid images from img_paths)
                        doc = save_new_cattle_to_db(cattle_id, cattle_name, cattle_class, img_paths, avg_embedding)
                        if doc:
                            st.success(f"✅ Successfully registered {cattle_name} ({cattle_class}) with ID {cattle_id}")
                            # Display uploaded images from DB
                            st.subheader("Registered Images:")
                            show_images_with_captions(doc["images"], title="Reference Images", from_db=True)
                            
                            # Update FAISS index immediately with new embedding
                            if avg_embedding is not None:
                                # Add to existing index
                                index.add(avg_embedding / np.linalg.norm(avg_embedding, axis=1, keepdims=True))
                                
                                # Update ordered IDs list in MongoDB
                                try:
                                    # Get current ordered IDs from MongoDB
                                    faiss_doc = faiss_index_collection.find_one({"_id": "faiss_index"})
                                    if faiss_doc and "ordered_ids" in faiss_doc:
                                        ordered_ids = faiss_doc["ordered_ids"]
                                    else:
                                        ordered_ids = []
                                    
                                    ordered_ids.append(cattle_id)
                                    
                                    # Save updated index and IDs to MongoDB
                                    save_faiss_to_mongodb(index, ordered_ids)
                                except:
                                    # If there's an issue, rebuild entirely
                                    rebuild_faiss()
                        else:
                            st.error("❌ Failed to save registration data")
                    else:
                        st.error("❌ No valid images found. Please upload different images with clear cattle muzzle visible.")
                        
                except Exception as e:
                    st.error(f"❌ Registration failed: {str(e)}")

# ================== TAB: Browse & Download ==================
with tabs[3]:
    st.header("📂 Browse & Download Registered Cattle")
    col1, col2, col3 = st.columns([3,3,2])
    with col1:
        q_id = st.text_input("Filter by ID (partial or full)", key="browse_id")
    with col2:
        q_name = st.text_input("Filter by Name (partial, case-insensitive)", key="browse_name")
    with col3:
        limit = st.number_input("Limit", min_value=1, max_value=500, value=50, step=1, key="browse_limit")

    if st.button("Search", key="browse_search"):
            docs = list_cattle_from_db(filter_id=q_id.strip() or None, filter_name=q_name.strip() or None, limit=int(limit))
            if not docs:
                st.info("No matching records found.")
            else:
                st.write(f"Found {len(docs)} records (showing up to {limit})")
                table_rows = []
                for d in docs:
                    table_rows.append({
                        "12_digit_id": d.get("12_digit_id"),
                        "cattle_name": d.get("cattle_name"),
                        "cattle_class": d.get("cattle_class"),
                        "created_at": d.get("created_at"),
                        "n_images": len(d.get("images", []))
                    })
                st.dataframe(pd.DataFrame(table_rows))

                ids = [d["12_digit_id"] for d in docs]
                selected = st.multiselect("Select specific IDs", options=ids)

                if selected:
                    download_docs = [get_cattle_by_id(s) for s in selected if get_cattle_by_id(s)]
                else:
                    download_docs = docs

                csv_bytes = create_metadata_csv_bytes(download_docs)
                zip_bytes = create_images_zip_bytes(download_docs)

                st.download_button("⬇️ Download metadata CSV", data=csv_bytes,
                                   file_name=f"cattle_metadata_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
                st.download_button("⬇️ Download images ZIP", data=zip_bytes,
                                   file_name=f"cattle_images_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip",
                                   mime="application/zip")

# ================== TAB: Quick Lookup ==================
with tabs[4]:
    st.header("🔎 Quick Lookup by ID")
    lookup_id = st.text_input("Enter exact 12-digit ID to view", key="lookup_id")
    if st.button("Lookup", key="lookup_button"):
        if not lookup_id:
            st.error("Enter an ID to lookup.")
        else:
            # Get from MongoDB
            doc = get_cattle_by_id(lookup_id.strip())
            
            if not doc:
                st.warning("No cattle found with that ID.")
            else:
                name = doc.get('cattle_name', doc.get('name', 'Unknown'))
                class_info = doc.get('cattle_class', doc.get('class', 'Unknown'))
                st.write(f"**{lookup_id} — {name} ({class_info})**")
                if 'created_at' in doc:
                    st.write(f"Created at: {doc['created_at']}")

                if 'images' in doc:
                    show_images_with_captions(doc["images"], title="Reference Images", from_db=True)

# ================== TAB: Management ==================
with tabs[5]:
    st.header("Manage Registered Cattle")
    
    # Get total cattle count from MongoDB
    total_cattle = 0
    try:
        total_cattle = cattle_collection.count_documents({})
        st.info(f"Managing {total_cattle} cattle records in MongoDB.")
    except Exception as e:
        st.error(f"MongoDB connection error: {e}")
    
    if total_cattle == 0:
        st.info("📝 No cattle registered yet.")
    else:
        # Search functionality
        search_term = st.text_input("🔍 Search cattle by ID, name, or class", placeholder="Enter search term...", key="mgmt_search")
        
        # MongoDB cattle management
        # Search in MongoDB
        if search_term:
            docs = list_cattle_from_db(filter_id=search_term, filter_name=search_term, limit=100)
        else:
            docs = list_cattle_from_db(limit=100)
        
        st.caption(f"Showing {len(docs)} cattle from MongoDB")
        
        for doc in docs:
                cattle_id = doc["12_digit_id"]
                name = doc["cattle_name"]
                class_info = doc.get("cattle_class", "Unknown")
                
                with st.expander(f"🆔 {cattle_id} - {name} ({class_info})"):
                    # Tab layout for better organization
                    edit_tabs = st.tabs(["📋 Details", "🖼️ Images", "🗑️ Delete"])
                    
                    with edit_tabs[0]:
                        st.subheader("Edit Details")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            new_name = st.text_input(
                                "Cattle Name", 
                                value=name, 
                                key=f"mongo_name_{cattle_id}"
                            )
                        with col2:
                            new_class = st.text_input(
                                "Cattle Class", 
                                value=class_info, 
                                key=f"mongo_class_{cattle_id}"
                            )
                        
                        if st.button(f"💾 Save Details", key=f"save_mongo_{cattle_id}"):
                            if new_name and new_class:
                                updates = {
                                    "cattle_name": new_name,
                                    "cattle_class": new_class
                                }
                                if update_cattle_in_db(cattle_id, updates):
                                    st.success("✅ Details updated successfully")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to update details")
                            else:
                                st.error("❌ Name and class cannot be empty")
                        
                        st.divider()
                        st.write(f"**ID:** {cattle_id}")
                        if "created_at" in doc:
                            st.write(f"**Created:** {doc['created_at']}")
                    
                    with edit_tabs[1]:
                        st.subheader("Manage Images")
                        
                        # Show existing images with remove option
                        current_images = doc.get("images", [])
                        if current_images:
                            st.write(f"**Current Images ({len(current_images)}):**")
                            cols = st.columns(min(3, len(current_images)))
                            for idx, img_data in enumerate(current_images):
                                with cols[idx % 3]:
                                    if img_data.get("b64"):
                                        try:
                                            img_bytes = base64.b64decode(img_data["b64"])
                                            if len(img_bytes) > 0:
                                                img = Image.open(io.BytesIO(img_bytes))
                                                st.image(img, caption=img_data.get("filename", f"Image {idx+1}"), use_column_width=True)
                                            else:
                                                st.error(f"❌ Empty image data: {img_data.get('filename', f'Image {idx+1}')}")
                                                continue
                                        except Exception as e:
                                            st.error(f"❌ Invalid image: {img_data.get('filename', f'Image {idx+1}')} - {str(e)[:50]}")
                                            continue
                                        if st.button(f"🗑️ Remove", key=f"remove_img_{cattle_id}_{idx}"):
                                            if delete_cattle_image_from_db(cattle_id, img_data["filename"]):
                                                st.success("✅ Image removed")
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to remove image")
                        else:
                            st.info("No images available")
                        
                        # Add new images
                        st.divider()
                        st.write("**Add New Images:**")
                        new_images = st.file_uploader(
                            "Upload additional images", 
                            type=["jpg","jpeg","png"], 
                            accept_multiple_files=True,
                            key=f"add_images_{cattle_id}"
                        )
                        
                        if new_images:
                            # Validate new images with YOLO
                            with st.spinner("Validating new images..."):
                                valid_images = []
                                invalid_count = 0
                                
                                for img_file in new_images:
                                    if roi_model:
                                        img = Image.open(img_file).convert("RGB")
                                        roi_results = roi_model.predict(img)
                                        if roi_results and len(roi_results[0].boxes) > 0:
                                            roi_box = max(roi_results[0].boxes, key=lambda b: b.conf)
                                            roi_conf = float(roi_box.conf)
                                            if roi_conf >= 0.60:
                                                valid_images.append(img_file)
                                            else:
                                                invalid_count += 1
                                        else:
                                            invalid_count += 1
                                    else:
                                        valid_images.append(img_file)
                                
                                if invalid_count > 0:
                                    st.warning(f"⚠️ {invalid_count} image(s) failed validation (ROI confidence < 0.60)")
                                
                                if valid_images and st.button(f"➕ Add {len(valid_images)} Valid Image(s)", key=f"confirm_add_{cattle_id}"):
                                    with st.spinner("Adding images..."):
                                        if add_cattle_images_to_db(cattle_id, valid_images):
                                            # Update embeddings if needed
                                            all_images = []
                                            updated_doc = get_cattle_by_id(cattle_id)
                                            if updated_doc and "images" in updated_doc:
                                                for img_data in updated_doc["images"]:
                                                    if img_data.get("b64"):
                                                        try:
                                                            img_bytes = base64.b64decode(img_data["b64"])
                                                            if len(img_bytes) > 0:
                                                                img = Image.open(io.BytesIO(img_bytes))
                                                                all_images.append(img)
                                                        except Exception as e:
                                                            st.warning(f"Skipping corrupted image: {img_data.get('filename', 'unknown')}")
                                                            continue
                                            
                                            if all_images:
                                                # Re-calculate embeddings for all images
                                                embeddings = []
                                                for img in all_images:
                                                    emb = embed_image(img)
                                                    if emb is not None:
                                                        embeddings.append(emb)
                                                
                                                if embeddings:
                                                    avg_embedding = np.mean(np.vstack(embeddings), axis=0, keepdims=True)
                                                    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding, axis=1, keepdims=True)
                                                    update_cattle_in_db(cattle_id, {"embedding": avg_embedding.flatten().tolist()})
                                                    rebuild_faiss()
                                            
                                            st.success(f"✅ Added {len(valid_images)} image(s) successfully")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to add images")
                    
                    with edit_tabs[2]:
                        st.subheader("Delete Cattle Record")
                        st.warning("⚠️ This action cannot be undone!")
                        st.write(f"This will permanently delete cattle **{name}** (ID: {cattle_id}) and all associated data.")
                        
                        if st.button(f"🗑️ Delete Permanently", key=f"delete_mongo_{cattle_id}", type="secondary"):
                            try:
                                cattle_collection.delete_one({"12_digit_id": cattle_id})
                                st.success("✅ Deleted from MongoDB successfully")
                                rebuild_faiss()
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting cattle: {str(e)}")

# ================== TAB: Database Viewer ==================
with tabs[6]:
    st.header("🗄️ Database Viewer")
    
    # MongoDB is connected
    st.success("✅ Connected to MongoDB")
    
    # Database statistics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        total_records = cattle_collection.count_documents({})
        with_embeddings = cattle_collection.count_documents({"embedding": {"$ne": None}})
        total_images = 0
        
        # Calculate total images
        for doc in cattle_collection.find({}, {"images": 1}):
            total_images += len(doc.get("images", []))
        
        avg_images = total_images / total_records if total_records > 0 else 0
        
        with col1:
            st.metric("Total Records", total_records)
        with col2:
            st.metric("With Embeddings", with_embeddings)
        with col3:
            st.metric("Total Images", total_images)
        with col4:
            st.metric("Avg Images/Cattle", f"{avg_images:.1f}")
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        
        st.divider()
        
        # View options
        view_tabs = st.tabs(["📊 Table View", "📄 Raw Data", "📈 Analytics", "🔧 Database Tools"])
        
        with view_tabs[0]:
            st.subheader("Table View")
            
            # Pagination
            items_per_page = st.select_slider("Items per page", options=[5, 10, 20, 50, 100], value=10)
            
            try:
                # Get all documents for table view
                all_docs = list(cattle_collection.find().sort("created_at", -1))
                
                if all_docs:
                    # Create table data
                    table_data = []
                    for doc in all_docs[:items_per_page]:
                        table_data.append({
                            "ID": doc["12_digit_id"],
                            "Name": doc["cattle_name"],
                            "Class": doc.get("cattle_class", "Unknown"),
                            "Images": len(doc.get("images", [])),
                            "Has Embedding": "✅" if doc.get("embedding") else "❌",
                            "Created": doc.get("created_at", "Unknown")[:10] if doc.get("created_at") else "Unknown"
                        })
                    
                    st.dataframe(table_data, use_container_width=True)
                    
                    if len(all_docs) > items_per_page:
                        st.caption(f"Showing {items_per_page} of {len(all_docs)} records")
                else:
                    st.info("No records found in database")
                    
            except Exception as e:
                st.error(f"Error loading table: {e}")
        
        with view_tabs[1]:
            st.subheader("Raw Database Records")
            
            # Record selector
            try:
                all_ids = [doc["12_digit_id"] for doc in cattle_collection.find({}, {"12_digit_id": 1})]
                
                if all_ids:
                    selected_id = st.selectbox("Select a record to view", all_ids)
                    
                    if selected_id:
                        doc = cattle_collection.find_one({"12_digit_id": selected_id})
                        if doc:
                            # Remove large binary data for display
                            display_doc = doc.copy()
                            if "_id" in display_doc:
                                display_doc["_id"] = str(display_doc["_id"])
                            
                            # Summarize images without showing base64
                            if "images" in display_doc:
                                image_summary = []
                                for img in display_doc["images"]:
                                    image_summary.append({
                                        "filename": img.get("filename", "Unknown"),
                                        "size": len(img.get("b64", "")) if "b64" in img else 0
                                    })
                                display_doc["images"] = image_summary
                            
                            # Summarize embedding
                            if "embedding" in display_doc and display_doc["embedding"]:
                                display_doc["embedding"] = f"Vector[{len(display_doc['embedding'])}]"
                            
                            st.json(display_doc)
                            
                            # Option to view full record
                            if st.checkbox("Show full record with base64 data", key=f"full_{selected_id}"):
                                st.warning("⚠️ Large amount of data below")
                                st.text(str(doc))
                else:
                    st.info("No records in database")
                    
            except Exception as e:
                st.error(f"Error viewing records: {e}")
        
        with view_tabs[2]:
            st.subheader("Database Analytics")
            
            try:
                # Class distribution
                class_dist = {}
                for doc in cattle_collection.find({}, {"cattle_class": 1}):
                    cattle_class = doc.get("cattle_class", "Unknown")
                    class_dist[cattle_class] = class_dist.get(cattle_class, 0) + 1
                
                if class_dist:
                    st.write("**Cattle Class Distribution:**")
                    for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- {class_name}: {count} cattle")
                
                st.divider()
                
                # Image statistics
                image_counts = []
                for doc in cattle_collection.find({}, {"images": 1}):
                    image_counts.append(len(doc.get("images", [])))
                
                if image_counts:
                    st.write("**Image Statistics:**")
                    st.write(f"- Min images per cattle: {min(image_counts)}")
                    st.write(f"- Max images per cattle: {max(image_counts)}")
                    st.write(f"- Average images: {sum(image_counts)/len(image_counts):.2f}")
                
                st.divider()
                
                # Embedding coverage
                total_recs = cattle_collection.count_documents({})
                with_emb = cattle_collection.count_documents({"embedding": {"$ne": None}})
                without_emb = total_recs - with_emb
                
                st.write("**Embedding Coverage:**")
                if total_recs > 0:
                    st.write(f"- With embeddings: {with_emb} ({with_emb/total_recs*100:.1f}%)")
                    st.write(f"- Without embeddings: {without_emb} ({without_emb/total_recs*100:.1f}%)")
                else:
                    st.write("- No records in database")
                
            except Exception as e:
                st.error(f"Error generating analytics: {e}")
        
        with view_tabs[3]:
            st.subheader("Database Tools")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export Data**")
                if st.button("📥 Export All Records as JSON"):
                    try:
                        all_docs = list(cattle_collection.find())
                        # Convert ObjectId to string for JSON serialization
                        for doc in all_docs:
                            if "_id" in doc:
                                doc["_id"] = str(doc["_id"])
                        
                        json_data = json.dumps(all_docs, indent=2)
                        st.download_button(
                            label="💾 Download JSON",
                            data=json_data,
                            file_name=f"cattle_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        st.success(f"✅ Prepared {len(all_docs)} records for download")
                    except Exception as e:
                        st.error(f"Error exporting data: {e}")
                
                if st.button("📥 Export Summary CSV"):
                    try:
                        import pandas as pd
                        
                        summary_data = []
                        for doc in cattle_collection.find():
                            summary_data.append({
                                "ID": doc["12_digit_id"],
                                "Name": doc["cattle_name"],
                                "Class": doc.get("cattle_class", "Unknown"),
                                "Images": len(doc.get("images", [])),
                                "Has_Embedding": bool(doc.get("embedding")),
                                "Created": doc.get("created_at", "")
                            })
                        
                        df = pd.DataFrame(summary_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"cattle_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        st.success(f"✅ Prepared summary of {len(summary_data)} records")
                    except Exception as e:
                        st.error(f"Error creating CSV: {e}")
            
            with col2:
                st.write("**Database Maintenance**")
                
                if st.button("🔄 Rebuild All Embeddings"):
                    if st.checkbox("I understand this will regenerate all embeddings", key="confirm_rebuild"):
                        with st.spinner("Rebuilding all embeddings..."):
                            updated_count = 0
                            for doc in cattle_collection.find():
                                if "images" in doc and doc["images"]:
                                    images = []
                                    for img_data in doc["images"]:
                                        if img_data.get("b64"):
                                            try:
                                                img_bytes = base64.b64decode(img_data["b64"])
                                                if len(img_bytes) > 0:
                                                    img = Image.open(io.BytesIO(img_bytes))
                                                    images.append(img)
                                            except Exception as e:
                                                st.warning(f"Skipping corrupted image in record {doc['12_digit_id']}")
                                                continue
                                    
                                    if images:
                                        # Calculate embeddings for all images
                                        embeddings = []
                                        for img in images:
                                            emb = embed_image(img)
                                            if emb is not None:
                                                embeddings.append(emb)
                                        
                                        if embeddings:
                                            # Calculate average embedding
                                            avg_embedding = np.mean(np.vstack(embeddings), axis=0, keepdims=True)
                                            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding, axis=1, keepdims=True)
                                            
                                            cattle_collection.update_one(
                                                {"_id": doc["_id"]},
                                                {"$set": {"embedding": avg_embedding.flatten().tolist()}}
                                            )
                                            updated_count += 1
                            
                            rebuild_faiss()
                            st.success(f"✅ Updated {updated_count} records with new embeddings")
                
                if st.button("🧹 Remove Records Without Images"):
                    if st.checkbox("I understand this will delete records", key="confirm_delete_no_images"):
                        result = cattle_collection.delete_many({"$or": [{"images": []}, {"images": {"$exists": False}}]})
                        st.success(f"✅ Deleted {result.deleted_count} records without images")
                        rebuild_faiss()
                        st.rerun()
                
                if st.button("🔍 Check Database Integrity"):
                    issues = []
                    for doc in cattle_collection.find():
                        # Check for missing fields
                        if "12_digit_id" not in doc:
                            issues.append(f"Document {doc.get('_id')} missing 12_digit_id")
                        if "cattle_name" not in doc:
                            issues.append(f"Document {doc.get('12_digit_id', 'unknown')} missing cattle_name")
                        
                        # Check ID format
                        if "12_digit_id" in doc and (len(doc["12_digit_id"]) != 12 or not doc["12_digit_id"].isdigit()):
                            issues.append(f"Invalid ID format: {doc['12_digit_id']}")
                    
                    if issues:
                        st.warning(f"⚠️ Found {len(issues)} issues:")
                        for issue in issues[:10]:  # Show first 10 issues
                            st.write(f"- {issue}")
                        if len(issues) > 10:
                            st.write(f"... and {len(issues) - 10} more")
                    else:
                        st.success("✅ Database integrity check passed!")


