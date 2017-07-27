
$('#chat-form').on('submit', function(event){
    event.preventDefault();

    $.ajax({
        url : '/post/',
        type : 'POST',
        data : { msgbox : $('#chat-msg').val()},

        success : function(json){
            $('#chat-msg').val('');

            $('#msg-list').append('<li class="clearfix">' +
                '<div class="message-data align-right">' +
                    '<span class="message-data-name">'+json.user+'</span>' +
                    '<img src="/static/img/user_avatar.jpg" alt="user" class="avatar"/>' +
                '</div>' +
                '<div class="float-right message you-message">' +
                    '<div>'+json.msg+'</div>'+
                    '<div class="legend"><legend class="message-data-time">'+json.created+'</legend></div>' +
                '</div></li>');

            $('#msg-list').append('<li class="clearfix">' +
                '<div class="message-data align-left">' +
                    '<img src="/static/img/deustche_bank_avatar.png" alt="db" class="avatar" />' +
                    '<span class="message-data-name"> KAMPS </span>' +
                '</div>' +
                '<div class="float-left message me-message">' +
                    '<div>'+json.rly+'</div>' +
                    '<div class="legend"><legend class="message-data-time">'+json.created+'</legend></div>' +
                '</div></li>');

            document.body.scrollTop = document.body.scrollHeight;
        }
    });
});

// function getMessages(){
//     if (!scrolling) {
//         $.get('/messages/', function(messages){
//             $('#msg-list').html(messages);
//             var chatlist = document.getElementById('msg-list-div');
//             chatlist.scrollTop = chatlist.scrollHeight;
//         });
//     }
//     scrolling = false;
// }
//
// var scrolling = false;
// $(function(){
//     $('#msg-list-div').on('scroll', function(){
//         scrolling = true;
//     });
//      refreshTimer = setInterval(getMessages, 1500);
// });

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