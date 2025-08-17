# -------------------------
# Imports and requirements
# -------------------------

from docarray import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import torch

# DocArray → Stores images, text, and embeddings in a structured way.
# CLIP → A model that links text and images (lets you search images using text).
# PIL → Handles opening/saving images.
# requests → Downloads images from the internet.
# io.BytesIO → Lets you treat downloaded bytes like a file for PIL to open.
# NumPy → For math on embeddings (vectors).
# PyTorch → Needed for the CLIP model.





# -------------------------
# Data (mini database)
# -------------------------
data = [
    {"text": "A fluffy white cat",
     "image_url": "https://static.wixstatic.com/media/e8ce1e_f0ed3d2309144b7db8ee6e1535014ff7~mv2.jpg/v1/fill/w_488,h_640,al_c,q_80,enc_auto/e8ce1e_f0ed3d2309144b7db8ee6e1535014ff7~mv2.jpg"},
    {"text": "A small brown dog",
     "image_url": "https://pet-health-content-media.chewy.com/wp-content/uploads/2024/09/11181403/202104iStock-1349456012.jpg"},
    {"text": "A ginger kitten playing",
     "image_url": "https://t3.ftcdn.net/jpg/14/52/38/32/360_F_1452383237_R0Je2ploBz5HmnWY212LoqFp7O0O8zpr.jpg"},
    {"text": "A black puppy running",
     "image_url": "https://i.pinimg.com/736x/96/56/e1/9656e1e7657586dc1b6835c5ca264812.jpg"},
]

#Takes an image URL. Downloads it with requests. If successful, converts it into an RGB PIL image.

def download_image(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            print(f"Failed {url} status {r.status_code}")
            return None
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


# Loops through your data. Downloads each image. Creates a DocArray Document: blob stores the actual image, tags stores extra info, here the "caption".
# Adds the document to the list docs.

docs = []
for item in data:
    img = download_image(item["image_url"])
    if not img:
        continue
    d = Document(blob=img)
    d.tags["caption"] = item["text"]
    docs.append(d)
print("Documents created:", len(docs))



# Loads the pre-trained CLIP model and its processor. The processor handles resizing, tokenizing, and normalizing input.
# .eval() puts the model in “read-only” inference mode.

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def tensor_to_np(tensor):
    return tensor.detach().cpu().numpy().squeeze()

def normalize_np(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x



# compute embeddings and store numpy arrays
# For each document:
# Process its caption → get text embedding.
# Process its image → get image embedding.
# Normalize both embeddings.
# Store them in tags for later comparison.

for doc in docs:
    cap = doc.tags["caption"]
    t_inputs = clip_processor(text=[cap], return_tensors="pt", padding=True)
    i_inputs = clip_processor(images=doc.blob, return_tensors="pt")

    with torch.no_grad():
        # text features
        if hasattr(clip_model, "get_text_features"):
            t_feat = clip_model.get_text_features(**t_inputs)
        else:
            out = clip_model(**t_inputs)
            t_feat = getattr(out, "text_embeds", None) or getattr(out, "last_hidden_state", None)
            if t_feat is None:
                raise RuntimeError("No text features available")
        # image features
        if hasattr(clip_model, "get_image_features"):
            i_feat = clip_model.get_image_features(**i_inputs)
        else:
            out = clip_model(**i_inputs)
            i_feat = getattr(out, "image_embeds", None) or getattr(out, "last_hidden_state", None)
            if i_feat is None:
                raise RuntimeError("No image features available")

    t_vec = normalize_np(tensor_to_np(t_feat))
    i_vec = normalize_np(tensor_to_np(i_feat))
    doc.tags["text_emb"] = t_vec.astype(np.float32)
    doc.tags["image_emb"] = i_vec.astype(np.float32)

print("Embeddings computed (normalized).")

# helper: compute scores and CLIP-probabilities
# Takes a query text (like "cute ginger kitten"). Turns it into a vector (q_vec). Measures cosine similarity with every stored image.
# Converts similarities to logits (scaled scores). Applies softmax → gives probabilities across all images.

def score_and_probs_for_query(query_text):
    q_inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        if hasattr(clip_model, "get_text_features"):
            q_feat = clip_model.get_text_features(**q_inputs)
        else:
            out = clip_model(**q_inputs)
            q_feat = getattr(out, "text_embeds", None) or getattr(out, "last_hidden_state", None)
            if q_feat is None:
                raise RuntimeError("No text features for query")
    q_vec = normalize_np(tensor_to_np(q_feat))

    # cosine scores (dot because normalized)
    cos_scores = np.array([float(np.dot(q_vec, doc.tags["image_emb"])) for doc in docs], dtype=np.float32)

    # CLIP uses a learned logit scale: logit_scale * (q · k)
    # get scalar and apply
    try:
        logit_scale = clip_model.logit_scale.exp().item()
    except Exception:
        logit_scale = 1.0
    logits = cos_scores * logit_scale

    # softmax probabilities across the candidates
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    return cos_scores, logits, probs

# run and print
query = "cute ginger kitten"
cos_scores, logits, probs = score_and_probs_for_query(query)

print("\nQuery:", query)
rows = []
for i, doc in enumerate(docs):
    caption = doc.tags.get("caption", "")
    print(f"{i+1}. caption: {repr(caption):40}  cosine: {cos_scores[i]:.4f}  logit: {logits[i]:.4f}  prob: {probs[i]:.4f}")

# save top image
top_idx = int(np.argmax(cos_scores))
docs[top_idx].blob.save("top_match.jpg")
print(f"\nTop match (by cosine) saved to top_match.jpg  index={top_idx} cosine={cos_scores[top_idx]:.4f} prob={probs[top_idx]:.4f}")
