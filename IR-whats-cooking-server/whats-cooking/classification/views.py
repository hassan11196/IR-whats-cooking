from django.shortcuts import render, HttpResponse
from django.views import View
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt


# Create your views here.

import os
from .helpers import build_knn_model, get_knn_label
from inverted_index.views import DocumentRetreival
from .models import ModelVectorSpace, DISTANCE_FORMULAS, KNNClasification

FILE_PATH = os.path.dirname(__file__) + '../../data/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model_name not being utilized rn, hardcoded for KNN


class Test(View):
    def get(self, request):
        return JsonResponse({'status': True, 'message': 'Server is up'}, status=200)


class BuildModel(View):
    def get(self, request, model_name):
        data = {}
        try:
            knn_model_obj = KNNClasification.objects.latest()
        except KNNClasification.DoesNotExist as e:
            return JsonResponse({'status': False, 'message': str(e),  'type': 'Unknown', 'data': data}, status=200)
        # vector_space = vsm_model_obj.data
        data = {
            # 'tf_func': str(vsm_model_obj.tf_func),
            # 'idf_func': str(vsm_model_obj.idf_func),
            # 'norm_func': str(vsm_model_obj.norm_func),
            # 'id': str(vsm_model_obj.id)
            'accuracy': knn_model_obj.accuracy,
            'train_size': knn_model_obj.train_size,
            'test_size': knn_model_obj.test_size,
            'dataset': knn_model_obj.dataset.dataset_name,
            'distance_formula': knn_model_obj.distance_formula,
            'id': knn_model_obj.id}
        return JsonResponse({'status': True, 'message': 'Model Status', 'data': data}, status=200)

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, model_name):
        options = request.POST

        status = build_knn_model(dataset=options['dataset'], train_size=float(options['train_size']),
                                 test_size=float(options['test_size']), k=int(options['k']), distance_formula=options['distance_formula'], re_index=options['re_index'])
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
                raise ValueError('Invalid Text Query')

            result = get_knn_label(
                query=request.POST['query'], k=int(request.POST['k']), dataset=request.POST['dataset'])
            # print(type(result))
            # print(result)

            # res = result.occurrance
            # doc_ids = list(map(lambda pos: pos,result.occurrance.keys()))

            return JsonResponse({'status': True, 'message': 'Query Result',  'type': 'label', 'result': result}, status=200)
            # else:
            #     return JsonResponse({'status': True, 'message': 'Something Went Wrong', 'result': list(result), 'type': 'Unknown'}, status=200)

        except BaseException as e:
            print(e)
            return JsonResponse({'status': False, 'message': str(e), 'result': ''}, status=400)
