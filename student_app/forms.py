from django import forms
from .models import StudentRecord

class StudentRecordForm(forms.ModelForm):
    class Meta:
        model = StudentRecord
        fields = [
            'study_hours_per_day',
            'extracurricular_hours_per_day',
            'sleep_hours_per_day',
            'social_hours_per_day',
            'physical_activity_hours_per_day',
            'stress_level',
        ]
        widgets = {
            'stress_level': forms.TextInput(attrs={'placeholder': 'ex: Low, Medium, High'}),
        }
