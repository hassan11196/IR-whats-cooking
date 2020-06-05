from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.views import View
from django.middleware.csrf import get_token
from django.views import View
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
import os
from .helpers import  build_index, get_vector_query
from .models import VectorSpaceModel, TF_CHOICES,IDF_CHOICES,NORM_CHOICES
from inverted_index.views import DocumentRetreival
FILE_PATH = os.path.dirname(__file__) + '../../data/' + 'Trump Speechs/speech_'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Test(View):
    def get(self, request):
        return JsonResponse({'status':True, 'message':'Server is up'}, status=200)


class Indexer(View):
    def get(self, request):
        vsm_model_obj = VectorSpaceModel.objects.latest('id')
        # vector_space = vsm_model_obj.data
        data = {
            'tf_func' : str(vsm_model_obj.tf_func),
            'idf_func' : str(vsm_model_obj.idf_func),
            'norm_func' : str(vsm_model_obj.norm_func),
            'id' : str(vsm_model_obj.id)
        }
        return JsonResponse({'status':True, 'message':'Indexer Status','data' : data}, status=200)

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request):
        options = request.POST
        status = build_index(tf_func = options['tf_func'], idf_func = options['idf_func'], norm_func = options['norm_func'])
        return JsonResponse({'status':status, 'message':'Starting Indexing'}, status=200)

class QueryEngine(View):
    def get(self, request):
        function_options = {
            'tf':TF_CHOICES, 'idf':IDF_CHOICES,
        }
        return JsonResponse({'status':True, 'message':'Query Engine Status', 'data':function_options}, status=200)

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request):
        try:
            result = []
            print(request.POST)
            if request.POST['query'] == '':
                raise ValueError('Invalid Query')
            
            result = get_vector_query(request.POST['query'], alpha=float(request.POST['alpha']))
            print(type(result))
            print(result)
            if(len(result['doc_ids']) == 0):
                raise ValueError('Term not in any Documents.')
            if isinstance(result, set):
                return JsonResponse({'status':True, 'message':'Query Result', 'result':list(result), 'type':'set', 'docs':list(result)}, status=200)
            if isinstance(result, dict):
                
                # res = result.occurrance
                # doc_ids = list(map(lambda pos: pos,result.occurrance.keys()))

                return JsonResponse({'status':True, 'message':'Query Result', 'result':result['occurrance'], 'type':'PostingList', 'docs':result['doc_ids']}, status=200)
            else:
                return JsonResponse({'status':True, 'message':'Something Went Wrong', 'result':list(result), 'type':'Unknown'}, status=200)
                

        except BaseException as e:
            print(e)
            return JsonResponse({'status':False, 'message':str(e), 'result' : ''}, status=400)