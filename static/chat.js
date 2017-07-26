$('#chat-form').on('submit', function(event){
    event.preventDefault();

    $.ajax({
        url : '/post/',
        type : 'POST',
        data : { msgbox : $('#chat-msg').val() },

        success : function(json){
            $('#chat-msg').val('');
            $('#msg-list').append('<li class="clearfix"><div class="message-data align-right"><span class="message-data-name"><i class="fa fa-circle you"></i> c.user.username </span> </div>');
            $('#msg-list').append('<div class="float-right message you-message">' + json.msg + '</div>'+'</li>');
            //jump to back end
            $('#msg-list').append('<li class="clearfix"><div class="message-data" ><span class="message-data-name" style="position:relative"><i class="fa fa-circle me"></i>KAMPS</span></div>');
            $('#msg-list').append('<div class="message me-message">Let me think about it....</div></li>');

            // var chatlist = document.getElementById('body');
            document.body.scrollTop = document.body.scrollHeight;
        }
    });
});

function getMessages(){
    if (!scrolling) {
        $.get('/messages/', function(messages){
            $('#msg-list').html(messages);
            var chatlist = document.getElementById('msg-list-div');
            chatlist.scrollTop = chatlist.scrollHeight;
        });
    }
    scrolling = false;
}

var scrolling = false;
$(function(){
    $('#msg-list-div').on('scroll', function(){
        scrolling = true;
    });
    refreshTimer = setInterval(getMessages, 500000);
});

$(document).ready(function() {
     $('#send').attr('disabled','disabled');
     $('#chat-msg').keyup(function() {
        if($(this).val() != '') {
           $('#send').removeAttr('disabled');
        }
        else {
        $('#send').attr('disabled','disabled');
        }
     });
 });

// using jQuery
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});