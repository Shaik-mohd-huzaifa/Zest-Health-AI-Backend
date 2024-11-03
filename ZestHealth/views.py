from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from Services.predication import prediction
import json

# @csrf_exempt
def ask(request):
    if request.method == "POST":
        data = json.loads(request.body.decode("utf-8"))
        prompt = data.get("user_input")

        intent = prediction(prompt)

    return JsonResponse({"response": intent})
