function methodShow(){
    var methodValue = document.getElementById('method').value;
    var lenFile = document.getElementById('upload-file').files.length ;
    
    if(lenFile == 0){
        alert('You have to input file');
        return false;
    }

    if(methodValue === 'clustering'){
        var k = document.getElementById('k').value;
        var maxIteration = document.getElementById('max-iteration').value;
        
        if (isNaN(k) == true || isNaN(maxIteration) == true) {
            alert('You have to input valid k and max iteration');
            return false;
        }else if (k.length == 0 || maxIteration.length == 0 ) {
            alert('You have to input k and max iteration');
            return false;
        }
    }

    const loading = document.createElement('div');
    loading.id = 'loading-screen';
    loading.insertAdjacentHTML('beforeend', '<i class="fas fa-circle-notch fa-spin"></i>');
    document.body.appendChild(loading);

    const form = document.forms["form-data"];
    const formData = new FormData(form);
    fetch('/code', {
        method: 'post',
        body: formData,
    })
    .then(response => {
        loading.remove();
        return response.json();
    })
    .then(result => {

        if (methodValue == 'statistics') {

            addTable(result.dataframe, document.querySelector('#data-head'));

            const descContainer = document.querySelector('#data-row-col');
            descContainer.innerHTML = '';
            const descText = document.createElement('p');
            descText.classList.add('center');
            descText.textContent = result.description;
            descContainer.appendChild(descText);
            
            addTable(result.summary, document.querySelector('#data-summary'));

            let hide = document.getElementById('stat-before-submit')
            let display = document.getElementById('stat-after-submit');

            hide.style.display = 'none';
            display.style.display = 'block';
            location.href = '#statistics';
        }
        else if (methodValue == 'classification') {
            if (result.message != undefined) {
                alert(result.message);
                return;
            }
            else {
                const accContainer = document.querySelector('#data-accuracy');
                accContainer.innerHTML = '';
                const accText = document.createElement('p');
                accText.classList.add('center');
                accText.textContent = result.accuracy;
                accContainer.appendChild(accText);

                addTable(result.prediction, document.querySelector('#data-prediction'));

                let hide = document.getElementById('class-before-submit')
                let display = document.getElementById('class-after-submit');

                hide.style.display = 'none';
                display.style.display = 'block';
                location.href = '#classification';
            }
        }
        else if (methodValue == 'clustering') {
            addTable(result.cluster, document.querySelector('#data-clustering'));

            addTable(result.centroid, document.querySelector('#cluster-centroid>.scrollable'));

            let hide = document.getElementById('cluster-before-submit')
            let display = document.getElementById('cluster-after-submit');

            hide.style.display = 'none';
            display.style.display = 'block';
            location.href = '#clustering';
        }

    })
    .catch(error => {
    });
}

function methodChange(source) {
    const target = document.getElementById('form-clustering');
    if (source.value !== 'clustering') target.style.display = 'none';
    else target.style.display = 'block';

}

function addTable(value, container) {
    container.scroll(0, 0);
    container.innerHTML = '';

    const fields = value.schema.fields.map(f => f.name);
    const data = value.data;

    const table = document.createElement('table');
    const tableHead = document.createElement('thead');
    const tableHeadRow = document.createElement('tr');
    fields.forEach(f => {
        let tableHeader = document.createElement('th');
        tableHeader.textContent = f;
        tableHeadRow.appendChild(tableHeader);
    });
    tableHead.appendChild(tableHeadRow);
    table.appendChild(tableHead);

    const tableBody = document.createElement('tbody');

    data.forEach(d => {
        let tableRow = document.createElement('tr');
        fields.forEach(f => {
            let tableData = document.createElement('td');
            tableData.textContent = d[f];
            tableRow.appendChild(tableData);
        });
        tableBody.appendChild(tableRow);
    });
    table.appendChild(tableBody);
    container.appendChild(table);
}