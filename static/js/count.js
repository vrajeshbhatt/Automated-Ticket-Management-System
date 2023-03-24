var count = {{count}}
if (count == 1) {
    $(".container").toggleClass("log-in");
    count = 0;
}
$(".container-form .btn2").click(function() {
    $(".container").toggleClass("log-in");
});