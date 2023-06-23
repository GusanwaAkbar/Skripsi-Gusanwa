from django.db import models

# Create your models here.

#Peruntukan,	Pusat_kota,	Visibilitas,	Bangunan,	Luas

from django.db import models

class TrainingData(models.Model):
    Peruntukan = models.CharField(max_length=255)
    Pusat_kota = models.CharField(max_length=255)
    Visibilitas = models.CharField(max_length=255)
    Bangunan = models.CharField(max_length=255)
    Luas = models.IntegerField()
    latitude = models.DecimalField(max_digits=20, decimal_places=12)
    longitude = models.DecimalField(max_digits=20, decimal_places=12)
    Deskripsi = models.CharField(max_length=255, null = True)

    def __str__(self):
        return self.nama

class TestingData(models.Model):
    Peruntukan = models.CharField(max_length=255)
    Pusat_kota = models.CharField(max_length=255)
    Visibilitas = models.CharField(max_length=255)
    Bangunan = models.CharField(max_length=255)
    Luas = models.IntegerField()
    latitude = models.DecimalField(max_digits=20, decimal_places=12)
    longitude = models.DecimalField(max_digits=20, decimal_places=12)
    Deskripsi = models.CharField(max_length=255, null = True)

class MyModel(models.Model):
    peruntukan = models.CharField(max_length=100)
    sentralitas = models.IntegerField()
    visibilitas = models.IntegerField()
    bangunan = models.CharField(max_length=100)
    luas = models.IntegerField()
    hasil = models.CharField(max_length=1000)

    def __str__(self):
        return self.nama