
function refresh() {
    var canvas = document.getElementById('paint');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}


function convertCanvasToImage() {
    var canvas = document.getElementById('paint');
	var image = new Image();
    //canvas.getContext("2d").fillStyle = "rgb(255,255,255)";
    //canvas.getContext("2d").fillRect(0, 0, canvas.width, canvas.height);
	image.src = canvas.toDataURL();
    console.log(image.src);
    jQuery.getJSON('/predict/?image="' + image.src + '"' , function(data) {
        document.getElementById('result').innerHTML = '';
        jQuery('#result').append('<p>' + data + '</p>');
        //console.log(data);
    });
}