import scipy
import pickle
from django.shortcuts import render, HttpResponse
from django.views import View
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# Create your views here.

import os
from .helpers import get_ml_label, JSONDocToVec, build_ml_model
from inverted_index.views import DocumentRetreival
from classification.models import ModelVectorSpace, DISTANCE_FORMULAS, KNNClasification, ClassificationMlModel

FILE_PATH = os.path.dirname(__file__) + '../../data/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Model_name not being utilized rn, hardcoded for KNN


class Test(View):
    def get(self, request):
        return JsonResponse({'status': True, 'message': 'Server is up'}, status=200)


class Ingredients(View):
    def get(self, request):
        limit = int(request.GET.get('limit', 0))
        offset = int(request.GET.get('offset', 0))

        mvs = ModelVectorSpace.objects.filter(
            dataset__dataset_name='whats-cooking').latest()
        dv = mvs.data

        length_ingredients = len(dv.ingredients)
        if (offset >= length_ingredients):
            return JsonResponse({'status': False, 'message': 'Offset Out Bound', 'data': []})
        if (offset+limit >= length_ingredients):
            ingredients = list(dv.ingredients)[offset:]
            return JsonResponse({'status': True, 'message': 'ingredients', 'data': ingredients})
        ingredients = list(dv.ingredients)[offset:offset+limit]

        return JsonResponse({'status': True, 'message': 'ingredients', 'data': ingredients})


class BuildModel(View):
    def get(self, request, model_name):
        data = {}
        try:
            model_obj = ClassificationMlModel.objects.filter(
                ml_model_type=model_name).latest()
            # knn_model_obj = KNNClasification.objects.latest()
        except ClassificationMlModel.DoesNotExist as e:
            return JsonResponse({'status': False, 'message': str(e),  'type': 'ClassificationMlModel.DoesNotExist', 'data': data}, status=200)
        # vector_space = vsm_model_obj.data
        data = {
            # 'tf_func': str(vsm_model_obj.tf_func),
            # 'idf_func': str(vsm_model_obj.idf_func),
            # 'norm_func': str(vsm_model_obj.norm_func),
            # 'id': str(vsm_model_obj.id)
            'accuracy': model_obj.accuracy,
            'train_size': model_obj.train_size,
            'test_size': model_obj.test_size,
            'dataset': model_obj.dataset.dataset_name,
            'distance_formula': model_obj.distance_formula,
            'id': model_obj.id}
        return JsonResponse({'status': True, 'message': 'Model Status', 'data': data}, status=200)

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, model_name):
        options = request.POST
        print(options)
        print(model_name)
        status = build_ml_model(dataset=options['dataset'], ml_model_type=model_name, train_size=float(options['train_size']),
                                test_size=float(options['test_size']), re_index=options['re_index'])
        return JsonResponse({'status': status, 'message': 'Model Trained'}, status=200)


class PredictionEngine(View):
    def get(self, request, model_name):
        function_options = {
            'distance_formula': DISTANCE_FORMULAS
        }
        return JsonResponse({'status': True, 'message': 'Prediction Engine Status', 'data': function_options}, status=200)

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, model_name):
        try:
            result = []
            # print(request.POST)
            if request.POST['query'] == '':
                raise ValueError('Invalid Ingredients Query')

            result = get_ml_label(
                query=request.POST['query'].split(','), dataset=request.POST['dataset'], ml_model_type=model_name)
            # print(type(result))
            # print(result)

            # res = result.occurrance
            # doc_ids = list(map(lambda pos: pos,result.occurrance.keys()))

            return JsonResponse({'status': True, 'message': 'Query Result',  'type': 'label', 'result': result[0]}, status=200)
            # else:
            #     return JsonResponse({'status': True, 'message': 'Something Went Wrong', 'result': list(result), 'type': 'Unknown'}, status=200)

        except BaseException as e:
            print(e)
            return JsonResponse({'status': False, 'message': str(e), 'result': ''}, status=400)
