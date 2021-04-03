function methodShow(){
    var methodValue = document.getElementById('method').value;
    var lenFile = document.getElementById('upload-file').files.length ;
    
    if(lenFile == 0){
        alert('You have to input file');
        return false;
    }

    if(methodValue == 'statistics'){
        var hide = document.getElementById('stat-before-submit')
        var display = document.getElementById('stat-after-submit');

        hide.style.display = 'none';
        display.style.display = 'block';
        location.href = '#statistics';
    } else if(methodValue == 'classification'){
        var hide = document.getElementById('class-before-submit')
        var display = document.getElementById('class-after-submit');

        hide.style.display = 'none';
        display.style.display = 'block';
        location.href = '#classification';
    }
}
