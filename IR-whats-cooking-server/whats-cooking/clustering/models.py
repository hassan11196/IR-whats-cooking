from django.db import models
from picklefield.fields import PickledObjectField
from classification.models import Dataset
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

DISTANCE_FORMULAS = [
    ('cosine_similarity', 'cosine_similarity'),
    ('euclidian_distance', 'euclidian_distance')
]

# Using Same Dataset and VectorSpace Model from Clustering


class KMeansClustering(models.Model):
    model = PickledObjectField()
    id = models.DateTimeField(auto_now_add=True, primary_key=True)
    dataset = models.ForeignKey(
        "classification.Dataset", verbose_name='Clustering Dataset', on_delete=models.CASCADE)
    vector_space = models.ForeignKey("classification.ModelVectorSpace",
                                     verbose_name="Clustering Vector Space", on_delete=models.CASCADE)
    train_size = models.FloatField(name='train_size', default=0.8)
    test_size = models.FloatField(name='test_size', default=0.2)
    cv_size = models.FloatField(name='cv_size', default=0)
    n_clusters = models.IntegerField(
        help_text='Number of Clusteres', name='n_clusters', default=3)
    distance_formula = models.CharField(
        name='distance_formula', choices=DISTANCE_FORMULAS, max_length=255)
    labels = PickledObjectField()
    file_names = PickledObjectField()
