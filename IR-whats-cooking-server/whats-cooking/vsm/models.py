from django.db import models

# Create your models here.
from picklefield.fields import PickledObjectField

TF_CHOICES = [
    ('normal', 'normal'),
    ('logarithm', 'logarithm'),
    ('augmented', 'augmented'),
    ('boolean', 'boolean'),
    ('log_ave', 'log_ave')
]
IDF_CHOICES = [
    ('no', 'no'),
    ('idf','idf'),
    ('prob_idf', 'prob_idf')
]
NORM_CHOICES = [
    ('none', 'none'),
    ('cosine', 'cosine')
]

class VectorSpaceModel(models.Model):
    data = PickledObjectField()
    status = models.BooleanField(default=False, name = 'status')
    id = models.DateTimeField(auto_now_add=True, primary_key = True)
    tf_func = models.CharField(name='tf_func', choices = TF_CHOICES, max_length=50)
    idf_func = models.CharField(name='idf_func', choices = IDF_CHOICES, max_length=50)
    norm_func = models.CharField(name='norm_func', choices = NORM_CHOICES, max_length=50)
