from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
	#return HttpResponse("Hello, World. You're here to view the Panama Papers visualization.")
	title = {'title': 'Dynamic Histograms'}
	return render(request, 'dynamic_histograms/index.html', title)