"""
URL configuration for skripsi_web project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from map import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.asset_map, name='asset_map'),

    path('sistem1', views.sistem1_view, name='sistem1_view'),
    path('sistem2', views.sistem2_view, name='sistem2_view'),
    path('sistem3', views.sistem3_view, name='sistem3_view'),
    path('sistem4', views.sistem4_view, name='sistem4_view'),
    path('sistem5', views.sistem5_view, name='sistem5_view'),
    path('predict', views.predict, name='predict'),
    
]

