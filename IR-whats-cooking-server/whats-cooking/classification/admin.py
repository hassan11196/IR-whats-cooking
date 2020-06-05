from django.contrib import admin

from .models import ModelVectorSpace, Dataset, KNNClasification
# Register your models here.
admin.site.register(Dataset)
admin.site.register(ModelVectorSpace)
admin.site.register(KNNClasification)
