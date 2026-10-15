import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import io
import base64
import json
import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.files.base import ContentFile

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite

from .models import UserProfile, Prediction


MODEL_PATH = os.path.join(settings.BASE_DIR, 'model.tflite')
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ FMD TFLite model loaded")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def run_inference(img):
    """Run model on a PIL image; return raw probability (0=FMD, 1=Healthy)."""
    arr = np.array(img.convert('RGB').resize((224, 224)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]['index'])[0][0])


def generate_heatmap(img):
    """
    Occlusion sensitivity heatmap (works with TFLite).
    Divides image into a 4x4 grid, masks each tile, and measures prediction change.
    Returns base64-encoded PNG.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    original = img.convert('RGB').resize((224, 224))
    base_prob = run_inference(original)

    grid = 4
    tile = 224 // grid
    sensitivity = np.zeros((grid, grid))

    for i in range(grid):
        for j in range(grid):
            masked = original.copy()
            m = np.array(masked)
            m[i*tile:(i+1)*tile, j*tile:(j+1)*tile] = 128
            masked = Image.fromarray(m)
            prob = run_inference(masked)
            sensitivity[i, j] = abs(base_prob - prob)

    # Normalize
    if sensitivity.max() > 0:
        sensitivity = sensitivity / sensitivity.max()

    # Upscale heatmap to full image size
    heatmap = np.kron(sensitivity, np.ones((tile, tile)))

    # Overlay on original
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(original)
    ax.imshow(heatmap, cmap='jet', alpha=0.45)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------
@csrf_exempt
@require_POST
def register(request):
    try:
        data = json.loads(request.body)
        username = data.get('username', '').strip()
        password = data.get('password', '')
        farm_name = data.get('farm_name', '').strip()
        district = data.get('district', '').strip()

        if not username or not password:
            return JsonResponse({'error': 'Username and password required'}, status=400)
        if len(password) < 6:
            return JsonResponse({'error': 'Password must be at least 6 characters'}, status=400)
        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already taken'}, status=400)

        user = User.objects.create_user(username=username, password=password)
        UserProfile.objects.create(user=user, farm_name=farm_name, district=district)
        login(request, user)
        return JsonResponse({'message': 'Account created', 'username': username})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
@require_POST
def login_view(request):
    try:
        data = json.loads(request.body)
        username = data.get('username', '').strip()
        password = data.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is None:
            return JsonResponse({'error': 'Invalid username or password'}, status=401)
        login(request, user)
        profile = getattr(user, 'userprofile', None)
        return JsonResponse({
            'message': 'Logged in',
            'username': user.username,
            'farm_name': profile.farm_name if profile else '',
            'district': profile.district if profile else '',
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
@require_POST
def logout_view(request):
    logout(request)
    return JsonResponse({'message': 'Logged out'})


@require_GET
def me(request):
    if not request.user.is_authenticated:
        return JsonResponse({'authenticated': False})
    profile = getattr(request.user, 'userprofile', None)
    return JsonResponse({
        'authenticated': True,
        'username': request.user.username,
        'farm_name': profile.farm_name if profile else '',
        'district': profile.district if profile else '',
    })


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
@csrf_exempt
@require_POST
def predict(request):
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image uploaded'}, status=400)
    try:
        img = Image.open(request.FILES['image'])
    except Exception as e:
        return JsonResponse({'error': f'Invalid image: {e}'}, status=400)

    prob = run_inference(img)
    result = 'FMD' if prob < 0.5 else 'Healthy'
    confidence = (1 - prob) if result == 'FMD' else prob

    if result == 'FMD':
        advice = (
            "⚠️ Foot and Mouth Disease detected.\n\n"
            "IMMEDIATE ACTIONS:\n"
            "1. Isolate this animal from the rest of the herd.\n"
            "2. Disinfect your hands, boots, and any equipment.\n"
            "3. Do NOT move any cattle off your farm.\n"
            "4. Provide soft feed and clean water.\n"
            "5. Call your nearest veterinary officer NOW.\n\n"
            "This is a notifiable disease — reporting is required by law in Eswatini."
        )
    else:
        advice = (
            "✅ The animal appears healthy.\n\n"
            "KEEP IT THAT WAY:\n"
            "1. Continue routine monitoring of all animals.\n"
            "2. Maintain clean water and feed.\n"
            "3. Follow your vaccination schedule.\n"
            "4. Limit visitors and disinfect footwear.\n"
            "5. Report any unusual signs immediately."
        )

    try:
        heatmap_b64 = generate_heatmap(img)
    except Exception as e:
        print(f"Heatmap failed: {e}")
        heatmap_b64 = None

    # Save if logged in
    if request.user.is_authenticated:
        request.FILES['image'].seek(0)
        pred = Prediction(
            user=request.user,
            result=result,
            confidence=round(confidence, 4),
            advice_given=advice,
        )
        pred.image.save(request.FILES['image'].name, request.FILES['image'], save=True)

    return JsonResponse({
        'result': result,
        'confidence': round(confidence, 4),
        'advice': advice,
        'heatmap': heatmap_b64,
    })


@require_GET
def history(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    preds = Prediction.objects.filter(user=request.user)[:20]
    data = [{
        'id': p.id,
        'result': p.result,
        'confidence': p.confidence,
        'created_at': p.created_at.strftime('%Y-%m-%d %H:%M'),
        'image_url': p.image.url if p.image else '',
    } for p in preds]
    return JsonResponse({'predictions': data})


# ------------------------------------------------------------------
# Learning content
# ------------------------------------------------------------------
LEARN_CONTENT = {
    "fmd": {
        "title": "Foot and Mouth Disease (FMD)",
        "subtitle": "Sifo lesikhulu (siSwati)",
        "sections": [
            {"heading": "What is FMD?",
             "body": "A highly contagious viral disease affecting cattle, sheep, goats, and pigs. It spreads very fast and can devastate a herd within days."},
            {"heading": "Symptoms",
             "body": "• Blisters/sores on tongue, gums, and mouth\n• Blisters on feet and between hooves\n• Excessive salivation (drooling)\n• Lameness and unwillingness to walk\n• High fever and reduced milk\n• Loss of appetite and weight loss"},
            {"heading": "How it spreads",
             "body": "• Direct contact between animals\n• Contaminated equipment, boots, clothing\n• Contaminated vehicles and feed\n• Airborne over short distances"},
            {"heading": "Prevention",
             "body": "• Vaccinate regularly (every 6 months)\n• Quarantine new animals for 30 days\n• Control farm visitors\n• Disinfect boots and vehicles\n• Report suspected cases immediately"},
            {"heading": "What to do if you suspect FMD",
             "body": "1. Isolate the sick animals IMMEDIATELY.\n2. Do NOT move cattle, milk, or meat off the farm.\n3. Disinfect your hands and equipment.\n4. Call your nearest veterinary officer.\n5. Keep a record of which animals are affected."}
        ]
    },
    "care": {
        "title": "Cattle Care Basics",
        "subtitle": "Keeping your herd healthy",
        "sections": [
            {"heading": "Daily care",
             "body": "• Fresh, clean water at all times\n• Balanced feed (grass, hay, supplements)\n• Salt licks and minerals\n• Clean, dry shelter\n• Regular exercise and grazing"},
            {"heading": "Routine health",
             "body": "• Vaccinate on schedule\n• Deworm every 3–6 months\n• Treat ticks and flies\n• Check hooves monthly\n• Watch for changes in appetite or behaviour"},
            {"heading": "Herd management",
             "body": "• Keep records of each animal\n• Isolate sick animals quickly\n• Breed at the right age and season\n• Provide shade in hot months\n• Prepare for dry season feed"},
            {"heading": "Signs of a healthy cow",
             "body": "• Bright, alert eyes\n• Shiny coat\n• Good appetite\n• Normal body temperature (38–39°C)\n• Active and moving freely"}
        ]
    },
    "biosecurity": {
        "title": "Biosecurity on Your Farm",
        "subtitle": "Stopping disease before it starts",
        "sections": [
            {"heading": "Access control",
             "body": "• Limit visitors to essential people only\n• Keep a visitor logbook\n• Provide footbaths at entrances\n• Park vehicles away from animal areas"},
            {"heading": "Animal movement",
             "body": "• Quarantine new animals for 30 days\n• Buy from trusted, disease-free sources\n• Keep records of all movements\n• Never mix new and old animals immediately"},
            {"heading": "Hygiene",
             "body": "• Disinfect equipment between uses\n• Wash hands and boots\n• Clean water troughs weekly\n• Remove manure regularly"},
            {"heading": "Reporting",
             "body": "• FMD is a NOTIFIABLE disease in Eswatini\n• Report immediately to your extension officer\n• Do not hide suspected cases — it protects everyone"}
        ]
    }
}


@require_GET
def learn_topics(request):
    return JsonResponse({'topics': [{'id': k, 'title': v['title']} for k, v in LEARN_CONTENT.items()]})


@require_GET
def learn_detail(request, topic):
    if topic not in LEARN_CONTENT:
        return JsonResponse({'error': 'Topic not found'}, status=404)
    return JsonResponse(LEARN_CONTENT[topic])


# ------------------------------------------------------------------
# AI Assistant (rule-based)
# ------------------------------------------------------------------
def assistant_response(query):
    q = query.lower()

    rules = [
        (['fever', 'hot', 'temperature'],
         "High fever in cattle is usually a sign of infection. Isolate the animal, provide shade and water, and take its temperature (normal is 38–39°C). If it stays above 40°C for more than 24 hours, call a vet."),
        (['saliva', 'drool', 'drooling', 'mouth'],
         "Excessive salivation or drooling, especially with sores in the mouth, is a classic sign of Foot and Mouth Disease. Isolate the animal immediately and contact your veterinary officer."),
        (['limp', 'lameness', 'foot', 'hoof', 'walk'],
         "Lameness can be caused by FMD, foot rot, or injuries. Check the hooves for sores or swelling. If accompanied by blisters or drooling, suspect FMD and isolate the animal right away."),
        (['lump', 'skin', 'node'],
         "Skin lumps are most often Lumpy Skin Disease (LSD). It spreads through flies and mosquitoes. Vaccination and vector control are the best prevention. Contact your vet for confirmation."),
        (['vaccine', 'vaccination', 'vaccinate'],
         "In Eswatini, FMD vaccination is recommended every 6 months. Follow the schedule from your veterinary extension officer. Keep records for every animal."),
        (['isolate', 'quarantine'],
         "Isolate any sick animal at least 20 metres from healthy ones. Use separate tools, boots, and clothing. Disinfect thoroughly. Never walk from sick to healthy animals without changing and disinfecting."),
        (['water', 'drink'],
         "Cattle need 30–50 litres of clean water per day. During hot weather or lactation, they may need up to 80 litres. Clean troughs weekly to prevent disease."),
        (['feed', 'food', 'grass', 'hay'],
         "A balanced diet includes grass or hay (bulk), supplemented with protein and minerals. Salt licks are essential. During dry season, store enough hay to last your herd."),
        (['pregnan', 'birth', 'calf', 'calving'],
         "Pregnant cows need extra nutrition and care, especially in the last 3 months. Watch for signs of calving and be ready to assist. Keep the calving area clean and dry."),
        (['tick', 'fly', 'mosquito', 'insect'],
         "Ticks spread diseases like heartwater and anaplasmosis. Dip or spray your cattle regularly. Control flies around the kraal to reduce Lumpy Skin Disease risk."),
        (['report', 'notify', 'officer', 'vet'],
         "In Eswatini, FMD is a notifiable disease. Report suspected cases IMMEDIATELY to your nearest veterinary extension officer. Do not move animals off your farm."),
        (['prevent', 'protect', 'biosecurity'],
         "Prevention is better than cure: vaccinate regularly, quarantine new animals, disinfect boots and equipment, limit visitors, and keep clean water and feed."),
    ]

    for keywords, response in rules:
        if any(k in q for k in keywords):
            return response

    return (
        "I'm your cattle health assistant. You can ask me about:\n\n"
        "• FMD symptoms and prevention\n"
        "• Fever, drooling, lameness, skin lumps\n"
        "• Vaccination schedules\n"
        "• Feeding and watering\n"
        "• Tick and fly control\n"
        "• Reporting to veterinary officers\n\n"
        "Try asking: 'My cow is drooling and has sores in the mouth' or 'How often should I vaccinate?'"
    )


@csrf_exempt
@require_POST
def assistant(request):
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        if not question:
            return JsonResponse({'error': 'No question provided'}, status=400)
        answer = assistant_response(question)
        return JsonResponse({'answer': answer})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def health(request):
    return JsonResponse({'status': 'ok'})