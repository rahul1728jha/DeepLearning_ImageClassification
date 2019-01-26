var imgId=1;

$(document).ready(function(){
        $('input[type="file"]').change(function(e){
			reset();
            var fileName = e.target.files[0].name;
            var size = Math.round(e.target.files[0].size / 1024) + " KB";
            var type = e.target.files[0].type;
            $('#imageName').append(fileName);
            $('#imageSize').append(size);
            $('#imageType').append(type);
            
			if (typeof(FileReader) != "undefined") 
			{
                var reader = new FileReader();
                reader.onload = function(e1) {
					var idName="#img"+window.imgId;
				  $(idName).attr("src",e1.target.result).addClass('borderClass').width('80%').height('80%');
					
			    }
               
                reader.readAsDataURL($(this)[0].files[0]);
            }
        });
    });

function callService()
{
	
	var urlToCall = "http://127.0.0.1:5000/predict";
	var idName1="#img"+window.imgId;
		$.ajax({
			type:'POST',
			url: urlToCall,
			data:$(idName1).attr("src"),
			success: function(data){
				console.log(data);
				var predictedClassName="#predictedClass"+window.imgId;
				$("#imagePrediction").append(data.FlowerType);
				$("#imageProbabilities").append(data.Probabilities);
				$(predictedClassName).append(data.FlowerType);
				window.imgId +=1
				if(window.imgId == 13)
				{
					window.imgId=1
				}
			}
		});  
}
function reset()
{
	$('#imageName').html('Name:');
	$('#imageSize').html('Size:');
	$('#imageType').html('Type:');
	$("#imagePrediction").html('Prediction: ');
	$("#imageProbabilities").html('Probabilities: ');
}

function refresh()
{
	window.location.reload();
}