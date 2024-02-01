from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import requests
import torch
from torchvision import transforms

app = FastAPI()

DETR_MODEL = "facebook/detr-resnet-50"
DETR_THRESHOLD = 0.5  # You can adjust the threshold as needed

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model = model.to(device).eval()

# Preprocess image
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# Perform object detection
def detect_objects(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

# Define FastAPI endpoint
@app.post("/detect")
async def detect_objects_endpoint(file: UploadFile = File(...)):
    try:
        image_tensor = preprocess_image(await file.read())
        outputs = detect_objects(image_tensor)
        
        # Filter detections based on confidence threshold
        filtered_detections = [
            {"label": int(label), "score": float(score),
             "box": [float(coord) for coord in box]}
            for label, score, box in zip(outputs["labels"][0].cpu().numpy(),
                                        outputs["scores"][0].cpu().numpy(),
                                        outputs["boxes"][0].cpu().numpy())
            if score > DETR_THRESHOLD
        ]

        return JSONResponse(content={"detections": filtered_detections})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application with UVicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
