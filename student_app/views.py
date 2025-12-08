
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import StudentRecordForm
from .models import StudentRecord
import joblib
import pandas as pd
import os

# charger le modèle une fois (chemin relatif à BASE_DIR)
MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_models", "gpa_predictor.joblib")
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)['pipeline']  # selon comment tu as sauvegardé
    return _model

def predict_gpa_from_form(cleaned_data):
    model = load_model()
    # prepare DataFrame avec mêmes colonnes que lors de l'entraînement
    df = pd.DataFrame([{
        'Study_Hours_Per_Day': cleaned_data['study_hours_per_day'],
        'Extracurricular_Hours_Per_Day': cleaned_data['extracurricular_hours_per_day'],
        'Sleep_Hours_Per_Day': cleaned_data['sleep_hours_per_day'],
        'Social_Hours_Per_Day': cleaned_data['social_hours_per_day'],
        'Physical_Activity_Hours_Per_Day': cleaned_data['physical_activity_hours_per_day'],
        'Stress_Level': cleaned_data['stress_level'],
    }])
    # NOTE: la pipeline attendait ces noms de colonnes ; adapte si nécessaire.
    pred = model.predict(df)[0]
    return float(pred)

def student_form_view(request):
    if request.method == 'POST':
        form = StudentRecordForm(request.POST)
        if form.is_valid():
            record = form.save(commit=False)
            # faire la prédiction
            try:
                record.predicted_gpa = predict_gpa_from_form(form.cleaned_data)
            except Exception as e:
                record.predicted_gpa = None
                print("Prediction error:", e)
            record.save()
            return render(request, "student_app/result.html", {"record": record})
    else:
        form = StudentRecordForm()
    return render(request, "student_app/form.html", {"form": form})

