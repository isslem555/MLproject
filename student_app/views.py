from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .forms import StudentRecordForm
from .models import StudentRecord

import joblib
import pandas as pd
import os
import json

# ==============================
# CHEMINS DES MODÈLES
# ==============================
BASE_DIR = settings.BASE_DIR
ML_MODELS_DIR = os.path.join(BASE_DIR, "ml_models")

STRESS_MODEL_PATH = os.path.join(ML_MODELS_DIR, "stress_model.pkl")
STRESS_SCALER_PATH = os.path.join(ML_MODELS_DIR, "scaler_stress.pkl")
STRESS_ENCODER_PATH = os.path.join(ML_MODELS_DIR, "stress_encoder.pkl")

stress_model = None
stress_scaler = None
stress_encoder = None

# ==============================
# CHARGER LES MODÈLES
# ==============================
def load_models_if_needed():
    global stress_model, stress_scaler, stress_encoder

    if stress_model is None:
        stress_model = joblib.load(STRESS_MODEL_PATH)
        stress_scaler = joblib.load(STRESS_SCALER_PATH)
        stress_encoder = joblib.load(STRESS_ENCODER_PATH)

# ==============================
# PRÉDICTION STRESS
# ==============================
def predict_stress(data):
    df = pd.DataFrame([[
        data["Study_Hours_Per_Day"],
        data["Extracurricular_Hours_Per_Day"],
        data["Sleep_Hours_Per_Day"],
        data["Social_Hours_Per_Day"],
        data["Physical_Activity_Hours_Per_Day"],
        data["GPA"]
    ]], columns=[
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA"
    ])

    df_scaled = stress_scaler.transform(df)
    pred_class = stress_model.predict(df_scaled)[0]
    pred_label = stress_encoder.inverse_transform([pred_class])[0]

    return pred_label

# ==============================
# 💡 CONSEILS AUTOMATIQUES
# ==============================
def get_stress_advice(stress_level):
    if stress_level == "High":
        return (
            "💙 Ton niveau de stress est élevé. "
            "Essaie de faire des pauses régulières, de respirer profondément "
            "et n’hésite pas à parler à quelqu’un de confiance."
        )

    elif stress_level == "Medium":
        return (
            "🙂 Ton stress est modéré. "
            "Une meilleure organisation de ton temps et un bon sommeil "
            "peuvent t’aider à t’améliorer."
        )

    elif stress_level == "Low":
        return (
            "🌟 Ton stress est faible. "
            "Continue avec ces bonnes habitudes, tu es sur la bonne voie !"
        )

    return "🤖 Prends soin de toi et écoute ton corps."

# ==============================
# VIEW FORMULAIRE PRINCIPALE
# ==============================
def student_form_view(request):
    form = StudentRecordForm(request.POST or None)
    record = None
    error_msg = None
    advice = None   # 👈 NOUVEAU

    if request.method == "POST" and form.is_valid():
        record = form.save(commit=False)

        try:
            load_models_if_needed()
        except Exception as e:
            error_msg = f"Erreur chargement modèles : {e}"

        if not error_msg:
            data_stress = {
                "Study_Hours_Per_Day": form.cleaned_data["study_hours_per_day"],
                "Extracurricular_Hours_Per_Day": form.cleaned_data["extracurricular_hours_per_day"],
                "Sleep_Hours_Per_Day": form.cleaned_data["sleep_hours_per_day"],
                "Social_Hours_Per_Day": form.cleaned_data["social_hours_per_day"],
                "Physical_Activity_Hours_Per_Day": form.cleaned_data["physical_activity_hours_per_day"],
                "GPA": 0
            }

            try:
                stress_label = predict_stress(data_stress)
                record.stress_level = stress_label

                # 💡 CONSEIL AUTOMATIQUE
                advice = get_stress_advice(stress_label)

            except Exception as e:
                error_msg = f"Erreur prédiction stress : {e}"

            record.save()

    return render(request, "student_app/form.html", {
        "form": form,
        "record": record,
        "error": error_msg,
        "advice": advice   # 👈 ENVOYÉ AU TEMPLATE
    })

# ==================================================
# 🤖 CHATBOT LOGIQUE
# ==================================================
def chatbot_response(message, stress_level):
    message = message.lower()

    if "stress" in message or "anxieux" in message:
        if stress_level == "High":
            return (
                "💙 Je comprends que tu te sentes stressé. "
                "Essaie de respirer profondément et de faire une petite pause. "
                "Tu n’es pas seul."
            )
        elif stress_level == "Medium":
            return (
                "🙂 Ton stress est modéré. "
                "Une bonne organisation et des pauses régulières peuvent t’aider."
            )
        else:
            return (
                "🌟 Ton stress est bas. "
                "Continue avec tes bonnes habitudes, tu es sur la bonne voie !"
            )

    if "dormir" in message or "fatigue" in message:
        return (
            "😴 Le sommeil est très important. "
            "Essaie de dormir 7 à 8 heures par nuit et évite les écrans avant de dormir."
        )

    if "conseil" in message or "aide" in message:
        return (
            "📘 Mon conseil : équilibre ton temps entre études, repos et loisirs. "
            "Chaque petit effort compte."
        )

    if "merci" in message:
        return "😊 Avec plaisir ! Je suis toujours là pour t’aider."

    return (
        "🤖 Je suis ton assistant bien-être. "
        "Tu peux me parler de ton stress, de ton sommeil ou demander des conseils."
    )

# ==================================================
# 🤖 API CHATBOT (AJAX)
# ==================================================
@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        message = data.get("message", "")
        stress = data.get("stress", "Unknown")

        reply = chatbot_response(message, stress)
        return JsonResponse({"reply": reply})
