# Generated by Django 4.2 on 2023-05-30 20:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('map', '0002_testingdata_deskripsi_trainingdata_deskripsi'),
    ]

    operations = [
        migrations.CreateModel(
            name='MyModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('peruntukan', models.CharField(max_length=100)),
                ('sentralitas', models.IntegerField()),
                ('visibilitas', models.IntegerField()),
                ('bangunan', models.CharField(max_length=100)),
                ('luas', models.IntegerField()),
                ('hasil', models.CharField(max_length=1000)),
            ],
        ),
    ]