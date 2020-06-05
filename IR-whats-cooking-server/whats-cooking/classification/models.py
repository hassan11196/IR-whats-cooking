from django.db import models
import datetime
from django.utils.timezone import now
from picklefield.fields import PickledObjectField
# Create your models here.
TF_CHOICES = [
    ('normal', 'normal'),
    ('logarithm', 'logarithm'),
    ('augmented', 'augmented'),
    ('boolean', 'boolean'),
    ('log_ave', 'log_ave')
]
IDF_CHOICES = [
    ('no', 'no'),
    ('idf', 'idf'),
    ('prob_idf', 'prob_idf')
]
NORM_CHOICES = [
    ('none', 'none'),
    ('cosine', 'cosine')
]

DATASET_LOCATION = [
    ('file_system', 'file_system'),
    ('database', 'database'),
    ('remote', 'remote')
]
DATASET_TYPES = [
    ('text_documents', 'text_documents'),
    ('json', 'json'),
    ('csv', 'csv'),
    ('images', 'images')
]
DISTANCE_FORMULAS = [
    ('cosine_similarity', 'cosine_similarity'),
    ('euclidian_distance', 'euclidian_distance')
]


class Dataset(models.Model):
    id = models.DateTimeField(
        auto_now_add=True, primary_key=True)
    name = models.CharField(name='dataset_name', max_length=50)
    type = models.CharField(name='dataset_type',
                            choices=DATASET_TYPES, default='text_documents', max_length=255)
    location = models.CharField(
        name='dataset_location', choices=DATASET_LOCATION, default='file_system', max_length=255)
    path = models.CharField(name='dataset_path', max_length=255)

    def __str__(self):
        return self.dataset_name

    class Meta:
        get_latest_by = ['id']


class ModelVectorSpace(models.Model):
    data = PickledObjectField()
    status = models.BooleanField(default=False, name='status')
    id = models.DateTimeField(
        auto_now_add=True, primary_key=True)
    dataset = models.ForeignKey(
        "classification.Dataset", verbose_name='Classification Dataset', on_delete=models.CASCADE)
    tf_func = models.CharField(
        name='tf_func', choices=TF_CHOICES, max_length=50, default='normal')
    idf_func = models.CharField(
        name='idf_func', choices=IDF_CHOICES, max_length=50, default='idf')
    norm_func = models.CharField(
        name='norm_func', choices=NORM_CHOICES, max_length=50)

    def __str__(self):
        return f'{self.dataset} - {self.id}'

    class Meta:
        get_latest_by = ['id']


class KNNClasification(models.Model):
    model = PickledObjectField()
    id = models.DateTimeField(auto_now_add=True, primary_key=True)
    time_created = models.DateTimeField(default=now)
    dataset = models.ForeignKey(
        "classification.Dataset", verbose_name='Classification Dataset', on_delete=models.CASCADE)
    vector_space = models.ForeignKey("classification.ModelVectorSpace",
                                     verbose_name="Classification Vector Space", on_delete=models.CASCADE)
    train_size = models.FloatField(name='train_size', default=0.8)
    test_size = models.FloatField(name='test_size', default=0.2)
    cv_size = models.FloatField(name='cv_size', default=0)
    k = models.IntegerField(help_text='K for K Neighbors', name='k', default=3)
    distance_formula = models.CharField(
        name='distance_formula', choices=DISTANCE_FORMULAS, max_length=255)
    accuracy = models.FloatField(name='accuracy', default=0)

    class Meta:
        get_latest_by = ['time_created']
