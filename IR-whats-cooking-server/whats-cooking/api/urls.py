from django.urls import path, include
from . import views

api = 'api'

urlpatterns = [
    path('test/', views.Test.as_view(), name='test'),
    path('ingredients', views.Ingredients.as_view(), name='ingredients'),
    path('model/<str:model_name>', views.BuildModel.as_view(), name='build_model'),
    path('predict/<str:model_name>',
         views.PredictionEngine.as_view(), name='prediction_engine'),
    path('docs/<str:dataset_name>',
         views.DocumentRetreival.as_view(), name='document_reteival'),
]
