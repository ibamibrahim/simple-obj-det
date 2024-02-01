from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import time


print('Device Count:', torch.cuda.device_count())
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

app = FastAPI()

print('Using Device:', device)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)

def get_objects(url):
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt").to(device)
    start = time.time()
    outputs = model(**inputs)
    end = time.time()
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    res = {}
    res['objects'] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        data = {
            'object_name': model.config.id2label[label.item()],
            'confidence': round(score.item(), 3),
            'location': box
        }
        res['objects'].append(data)

    res['latency'] = end - start
    res['device'] = str(model.device)
    return res

@app.get("/detect")
def detect(url: str):
    return JSONResponse(get_objects(url))

# Run the application with UVicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
