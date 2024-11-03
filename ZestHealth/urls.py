from django.urls import path
from ZestHealth import views


urlpatterns = [path("ask", views.ask, name="Ask ChatBot")]
