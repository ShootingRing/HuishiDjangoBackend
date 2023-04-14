from django.db import models


# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=16)
    password = models.CharField(max_length=16)
    id = models.IntegerField(primary_key=True)
    token = models.CharField(max_length=255, default='', null=True)
