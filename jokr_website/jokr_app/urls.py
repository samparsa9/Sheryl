# jokr_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('data/', views.get_data, name='get_data'),
]
