$(document).ready(function () {
    $('#draggableBtn').on('mousedown', function (e) {
        var offsetX = e.clientX - $(this).offset().left;
        var offsetY = e.clientY - $(this).offset().top;
        $(document).on('mousemove', function (e) {
            $('.draggable').offset({
                top: e.clientY - offsetY,
                left: e.clientX - offsetX
            });
        });
        $(document).on('mouseup', function () {
            $(document).off('mousemove');
        });
    });
});