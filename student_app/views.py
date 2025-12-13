from django.shortcuts import render
from django.conf import settings
from .forms import StudentRecordForm
from .models import StudentRecord
import joblib
import pandas as pd
import os

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
    # Créer DataFrame avec les données (avec GPA factice)
    df = pd.DataFrame([[
        data["Study_Hours_Per_Day"],
        data["Extracurricular_Hours_Per_Day"],
        data["Sleep_Hours_Per_Day"],
        data["Social_Hours_Per_Day"],
        data["Physical_Activity_Hours_Per_Day"],
        data["GPA"]  # colonne factice nécessaire pour le modèle existant
    ]], columns=[
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA"
    ])

    # Mise à l’échelle
    df_scaled = stress_scaler.transform(df)

    # Prédiction
    pred_class = stress_model.predict(df_scaled)[0]
    pred_label = stress_encoder.inverse_transform([pred_class])[0]

    return pred_label

# ==============================
# VIEW FORMULAIRE
# ==============================
def student_form_view(request):
    form = StudentRecordForm(request.POST or None)
    record = None
    error_msg = None

    if request.method == "POST" and form.is_valid():
        record = form.save(commit=False)

        try:
            load_models_if_needed()
        except Exception as e:
            error_msg = f"Erreur chargement modèles : {e}"

        if not error_msg:
            # Préparer les données pour le modèle
            data_stress = {
                "Study_Hours_Per_Day": form.cleaned_data["study_hours_per_day"],
                "Extracurricular_Hours_Per_Day": form.cleaned_data["extracurricular_hours_per_day"],
                "Sleep_Hours_Per_Day": form.cleaned_data["sleep_hours_per_day"],
                "Social_Hours_Per_Day": form.cleaned_data["social_hours_per_day"],
                "Physical_Activity_Hours_Per_Day": form.cleaned_data["physical_activity_hours_per_day"],
                "GPA": 0  # colonne factice
            }

            try:
                # Prédiction du stress
                stress_label = predict_stress(data_stress)
                record.stress_level = stress_label
            except Exception as e:
                error_msg = f"Erreur prédiction stress : {e}"

            # Sauvegarder l'enregistrement
            record.save()

    return render(request, "student_app/form.html", {
        "form": form,
        "record": record,
        "error": error_msg
    })
