from django.shortcuts import render
from django.contrib.auth import authenticate, logout, login
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from chat_app import settings

from .models import Chat
from .test import hello
from datetime import datetime
# import moment

def Login(request):
    next = request.GET.get('next', '/home/')
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)
                return HttpResponseRedirect(next)
            else:
                return HttpResponse("Account is not active at the moment.")
        else:
            return HttpResponseRedirect(settings.LOGIN_URL)
    return render(request, "alpha/login.html", {'next': next})

def Logout(request):
    logout(request)
    return HttpResponseRedirect('/login/')

def Home(request):
    c = Chat.objects.all()
    # return render(request, "alpha/home.html", {'home': 'active', 'chat': c})
    return render(request, "beta/kamps.html", {'home': 'active', 'chat': c})

def Post(request):

    if request.method == "POST":
        msg = request.POST.get('msgbox', None)
        returner = hello(msg)
        now=datetime.today()
        nowC= datetime.strftime(now,"%B %d, %Y, %I:%M %P")

        c = Chat(user=request.user, message=msg, reply=returner, created=now)
        if msg != '':
            c.save()
        return JsonResponse({ 'msg': msg, 'user': c.user.username, 'rly':returner, 'created':nowC })
    else:
        return HttpResponse('Request must be POST.')


def Messages(request):
    c = Chat.objects.all()
    return render(request, 'alpha/messages.html', {'chat': c})

# def Tester(request):
#     returner = hello()
#     return JsonResponse({'tester':returner})