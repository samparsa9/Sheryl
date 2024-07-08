from django.shortcuts import render
from django.http import JsonResponse
from django.db import connection
from .models import PortfolioOverview

# Create your views here.
# def get_data(request):
#     cursor = connection.cursor()
#     cursor.execute("SELECT * FROM portfolio_positions LIMIT 10")
#     data = cursor.fetchall()
# 
#     return JsonResponse(data, safe=False)
def get_data(request):
    data = list(PortfolioOverview.objects.all().values())
    return JsonResponse(data, safe=False)

def index(request):
    return render(request, 'index.html')