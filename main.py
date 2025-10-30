from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import base64
import json
from io import BytesIO
import uvicorn

app = FastAPI()

# Allow requests from anywhere (we'll use your Vercel URL later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def decode_image(file_bytes):
    """Convert uploaded file to PIL Image"""
    img = Image.open(BytesIO(file_bytes))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def pil_to_array(img):
    """Convert PIL Image to numpy array"""
    return np.array(img)

def array_to_pil(arr):
    """Convert numpy array to PIL Image"""
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

def encode_image(img):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    if isinstance(img, np.ndarray):
        img = array_to_pil(img)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def remove_background_simple(img):
    """Simple background removal"""
    img_array = pil_to_array(img)
    gray = img.convert('L')
    gray_array = pil_to_array(gray)
    threshold = 50
    mask = gray_array > threshold
    result = img_array.copy()
    for i in range(3):
        result[:, :, i] = result[:, :, i] * mask
    return array_to_pil(result)

def simple_align(img1, img2):
    """Align images by resizing"""
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.Resampling.LANCZOS)
    return img1

def detect_edges(img):
    """Edge detection using Sobel"""
    img_array = pil_to_array(img.convert('L'))
    sx = ndimage.sobel(img_array, axis=0, mode='constant')
    sy = ndimage.sobel(img_array, axis=1, mode='constant')
    edges = np.hypot(sx, sy)
    edges = (edges / edges.max() * 255).astype(np.uint8)
    edges = (edges > 50).astype(np.uint8) * 255
    return edges

def compute_difference(img1, img2, use_edge_detection=False):
    """Compute difference between images"""
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.Resampling.LANCZOS)
    
    gray1 = pil_to_array(img1.convert('L')).astype(float) / 255.0
    gray2 = pil_to_array(img2.convert('L')).astype(float) / 255.0
    pixel_diff = np.abs(gray1 - gray2)
    ssim_score, ssim_map = ssim(gray1, gray2, full=True, data_range=1.0)
    ssim_diff = 1 - ssim_map
    
    if use_edge_detection:
        edges1 = detect_edges(img1)
        edges2 = detect_edges(img2)
        edge_diff = np.abs(edges1.astype(float) - edges2.astype(float)) / 255.0
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        edge_density1 = ndimage.convolve(edges1.astype(float) / 255.0, kernel)
        edge_density2 = ndimage.convolve(edges2.astype(float) / 255.0, kernel)
        density_diff = np.abs(edge_density1 - edge_density2)
        difference_map = (0.2 * pixel_diff + 0.2 * ssim_diff + 0.3 * edge_diff + 0.3 * density_diff)
    else:
        difference_map = (pixel_diff + ssim_diff) / 2
    
    return difference_map, float(ssim_score)

def find_contours(binary_image):
    """Find contours using connected components"""
    labeled, num_features = ndimage.label(binary_image)
    contours = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            w = int(x_max - x_min)
            h = int(y_max - y_min)
            area = float(len(coords))
            contours.append({
                'bbox': [int(x_min), int(y_min), w, h],
                'area': area,
                'aspect_ratio': float(w / h if h > 0 else 0)
            })
    return contours

def is_text_region(change):
    """Check if region is likely text"""
    area = change['area']
    aspect_ratio = change['aspect_ratio']
    is_elongated = aspect_ratio > 2.5 or aspect_ratio < 0.4
    is_small_medium = 100 < area < 5000
    return is_elongated and is_small_medium

def find_changes(difference_map, sensitivity, filter_text=False):
    """Find changed regions"""
    binary = (difference_map > sensitivity).astype(np.uint8) * 255
    from scipy.ndimage import binary_closing, binary_opening
    struct = np.ones((5, 5))
    binary = binary_closing(binary > 0, structure=struct).astype(np.uint8) * 255
    binary = binary_opening(binary > 0, structure=struct).astype(np.uint8) * 255
    changes = find_contours(binary)
    changes = [c for c in changes if c['area'] > 100]
    if filter_text:
        changes = [c for c in changes if not is_text_region(c)]
    changes.sort(key=lambda c: c['area'], reverse=True)
    return changes, binary

def create_visualizations(img2, difference_map, changes):
    """Create result images"""
    diff_normalized = (difference_map * 255).astype(np.uint8)
    diff_colored = np.zeros((*difference_map.shape, 3), dtype=np.uint8)
    
    for i in range(diff_normalized.shape[0]):
        for j in range(diff_normalized.shape[1]):
            val = diff_normalized[i, j]
            if val < 64:
                diff_colored[i, j] = [0, 0, val * 4]
            elif val < 128:
                diff_colored[i, j] = [0, (val - 64) * 4, 255]
            elif val < 192:
                diff_colored[i, j] = [(val - 128) * 4, 255, 255 - (val - 128) * 4]
            else:
                diff_colored[i, j] = [255, 255 - (val - 192) * 4, 0]
    
    diff_img = array_to_pil(diff_colored)
    result_img = img2.copy()
    draw = ImageDraw.Draw(result_img)
    
    for i, change in enumerate(changes[:10]):
        x, y, w, h = change['bbox']
        color = (255, 0, 0) if i == 0 else (0, 255, 0)
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
        text = f"#{i+1}"
        draw.text((x, max(0, y-15)), text, fill=color)
    
    return diff_img, result_img

@app.post("/analyze")
async def analyze_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    config: str = Form(...)
):
    try:
        config_dict = json.loads(config)
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        img1 = decode_image(img1_bytes)
        img2 = decode_image(img2_bytes)
        
        max_dim = 800
        if max(img1.size) > max_dim:
            ratio = max_dim / max(img1.size)
            new_size = tuple(int(dim * ratio) for dim in img1.size)
            img1 = img1.resize(new_size, Image.Resampling.LANCZOS)
        
        if max(img2.size) > max_dim:
            ratio = max_dim / max(img2.size)
            new_size = tuple(int(dim * ratio) for dim in img2.size)
            img2 = img2.resize(new_size, Image.Resampling.LANCZOS)
        
        if config_dict.get('remove_background', False):
            img1 = remove_background_simple(img1)
            img2 = remove_background_simple(img2)
        
        img1 = simple_align(img1, img2)
        difference_map, ssim_score = compute_difference(img1, img2, config_dict.get('use_edge_detection', False))
        changes, binary = find_changes(difference_map, config_dict.get('sensitivity', 0.15), config_dict.get('filter_text_regions', False))
        diff_img, annotated_img = create_visualizations(img2, difference_map, changes)
        diff_base64 = encode_image(diff_img)
        annotated_base64 = encode_image(annotated_img)
        
        return JSONResponse({
            "success": True,
            "changes_count": len(changes),
            "ssim_score": ssim_score,
            "changes": changes[:10],
            "difference_map": diff_base64,
            "annotated_image": annotated_base64
        })
    except Exception as e:
        import traceback
        return JSONResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)

@app.get("/")
async def root():
    return {"message": "FrameShift API v1.1", "status": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)