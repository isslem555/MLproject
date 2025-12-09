from django.shortcuts import render
from django.conf import settings
from .forms import StudentRecordForm
from .models import StudentRecord
import joblib
import pandas as pd
import os

# Chemins vers les fichiers ML
STRESS_MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_models", "stress_model.pkl")
STRESS_SCALER_PATH = os.path.join(settings.BASE_DIR, "ml_models", "scaler_stress.pkl")
STRESS_ENCODER_PATH = os.path.join(settings.BASE_DIR, "ml_models", "stress_encoder.pkl")

GPA_MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_models", "gpa_predictor.pkl")
GPA_SCALER_PATH = os.path.join(settings.BASE_DIR, "ml_models", "gpa_scaler.pkl")

# Variables globales pour éviter de recharger plusieurs fois
stress_model = None
stress_scaler = None
stress_encoder = None
gpa_model = None
gpa_scaler = None

def load_models_if_needed():
    """Charge les modèles et transformateurs si ce n'est pas déjà fait."""
    global stress_model, stress_scaler, stress_encoder, gpa_model, gpa_scaler

    if stress_model is None:
        if not all(os.path.exists(p) for p in [STRESS_MODEL_PATH, STRESS_SCALER_PATH, STRESS_ENCODER_PATH]):
            raise FileNotFoundError("Fichier de modèle ou transformateur de stress introuvable !")
        stress_model = joblib.load(STRESS_MODEL_PATH)
        stress_scaler = joblib.load(STRESS_SCALER_PATH)
        stress_encoder = joblib.load(STRESS_ENCODER_PATH)

    if gpa_model is None:
        if not all(os.path.exists(p) for p in [GPA_MODEL_PATH, GPA_SCALER_PATH]):
            raise FileNotFoundError("Fichier de modèle ou transformateur de GPA introuvable !")
        gpa_model = joblib.load(GPA_MODEL_PATH)
        gpa_scaler = joblib.load(GPA_SCALER_PATH)

def predict_stress(data):
    """Prédit le niveau de stress à partir des données brutes."""
    df = pd.DataFrame([data])
    df_scaled = stress_scaler.transform(df)
    pred_class = stress_model.predict(df_scaled)[0]
    pred_label = stress_encoder.inverse_transform([pred_class])[0]
    return pred_label

def predict_gpa(data, stress_label):
    """Prédit le GPA à partir des données et du stress prédit."""
    stress_encoded = stress_encoder.transform([stress_label])[0]
    df = pd.DataFrame([{
        **data,
        'Stress_Level': stress_encoded
    }])
    df_scaled = gpa_scaler.transform(df)
    gpa_pred = gpa_model.predict(df_scaled)[0]
    return round(float(gpa_pred), 2)

def student_form_view(request):
    """Affiche le formulaire et prédictions de stress et GPA après soumission."""
    form = StudentRecordForm(request.POST or None)
    record = None
    error_msg = None

    if request.method == 'POST' and form.is_valid():
        record = form.save(commit=False)

        # Charger les modèles
        try:
            load_models_if_needed()
        except Exception as e:
            error_msg = f"Erreur chargement modèles : {e}"

        if not error_msg:
            # Préparer les données pour la prédiction
            data = {
                'study_hours_per_day': form.cleaned_data['study_hours_per_day'],
                'sleep_hours_per_day': form.cleaned_data['sleep_hours_per_day'],
                'social_hours_per_day': form.cleaned_data['social_hours_per_day'],
                'physical_activity_hours_per_day': form.cleaned_data['physical_activity_hours_per_day'],
            }

            # Prédiction du stress
            try:
                stress_label = predict_stress(data)
                record.predicted_stress = stress_label
            except Exception as e:
                record.predicted_stress = None
                error_msg = f"Erreur prédiction stress : {e}"

            # Prédiction du GPA seulement si stress réussi
            if record.predicted_stress and not error_msg:
                try:
                    gpa_pred = predict_gpa(data, record.predicted_stress)
                    record.predicted_gpa = gpa_pred
                except Exception as e:
                    record.predicted_gpa = None
                    error_msg = f"Erreur prédiction GPA : {e}"
            else:
                record.predicted_gpa = None

            record.save()

    context = {
        "form": form,
        "record": record,
        "error": error_msg
    }
    return render(request, "student_app/form.html", context)
