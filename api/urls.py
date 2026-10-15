from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health, name='health'),

    # Auth
    path('auth/register/', views.register, name='register'),
    path('auth/login/', views.login_view, name='login'),
    path('auth/logout/', views.logout_view, name='logout'),
    path('auth/me/', views.me, name='me'),

    # Prediction & history
    path('predict/', views.predict, name='predict'),
    path('history/', views.history, name='history'),

    # Learning
    path('learn/', views.learn_topics, name='learn_topics'),
    path('learn/<str:topic>/', views.learn_detail, name='learn_detail'),

    # Assistant
    path('assistant/', views.assistant, name='assistant'),
]