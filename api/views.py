import os
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Disable GPU / CUDA lookup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow warnings

import tensorflow as tf
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

MODEL_PATH = os.path.join(settings.BASE_DIR, 'mobileNetV2_FMD.h5')
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ FMD model loaded")

def preprocess(img):
    img = img.convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@csrf_exempt
@require_POST
def predict(request):
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image uploaded'}, status=400)
    try:
        img = Image.open(request.FILES['image'])
    except Exception as e:
        return JsonResponse({'error': f'Invalid image: {e}'}, status=400)

    arr = preprocess(img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    result = 'FMD' if prob < 0.5 else 'Healthy'
    confidence = prob if result == 'FMD' else 1 - prob
    advice = (
        "FMD detected. Isolate the animal. Disinfect. Call a vet."
        if result == 'FMD' else
        "Healthy. Continue routine monitoring."
    )
    return JsonResponse({
        'result': result,
        'confidence': round(confidence, 4),
        'advice': advice,
    })

def health(request):
    return JsonResponse({'status': 'ok'})