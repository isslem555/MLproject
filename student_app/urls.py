from django.urls import path
from . import views

urlpatterns = [
    path('', views.student_form_view, name='student_form'),  # page principale du formulaire
    path('predict/', views.student_form_view, name='student_predict'),  # optionnel
    path('chatbot/', views.chatbot_api, name='chatbot_api'),  # 🤖 chatbot
]
