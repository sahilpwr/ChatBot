from django.db import models
from django.contrib.auth.models import User
from datetime import datetime

class Chat(models.Model):
    created = models.DateTimeField(default=datetime.today,blank=True, null=True)
    user = models.ForeignKey(User)
    message = models.CharField(max_length=200)
    reply= models.CharField(max_length=200, default='')

    def __unicode__(self):
        return self.message
