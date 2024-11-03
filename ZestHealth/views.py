from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


# @csrf_exempt
def ask(request):
    print(request.method)
    return JsonResponse({"response": "Ask what you want"})
