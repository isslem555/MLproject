from django.db import models


class StudentRecord(models.Model):
    # fields correspondant au dataset (sauf Student_ID qui peut être auto)
    study_hours_per_day = models.FloatField()
    extracurricular_hours_per_day = models.FloatField()
    sleep_hours_per_day = models.FloatField()
    social_hours_per_day = models.FloatField()
    physical_activity_hours_per_day = models.FloatField()

    stress_level = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Record {self.id} - stress: {self.stress_level}"
