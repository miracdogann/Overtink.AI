from django.urls import path
from . import views 

urlpatterns = [
    path('',views.ped_api ,name="ped_api"),
    path('predict/', views.predict_image_api  , name='predict'),

]