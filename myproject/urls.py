from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('student/', include('student_app.urls')),  # déjà ok
    path('', RedirectView.as_view(url='/student/')),  # redirige / vers /student/
]
