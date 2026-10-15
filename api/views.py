import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# Use TFLite runtime (tiny, fast)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite

MODEL_PATH = os.path.join(settings.BASE_DIR, 'model.tflite')

# Load the model ONCE at startup
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ FMD TFLite model loaded")


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
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

    # LABELS ARE FLIPPED: 0 = FMD, 1 = Healthy
    result = 'FMD' if prob < 0.5 else 'Healthy'
    confidence = (1 - prob) if result == 'FMD' else prob
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