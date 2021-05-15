from bottle import route, run, static_file, request

from io import BytesIO
from code import summary_data, check_independent_var, data_shape, naive_bayes, print_naive_bayes_accuracy, kmeans
from pandas import read_csv
import os

@route('/')
def html():
    return static_file('index.html', root='.')

@route('/css/<filepath:path>')
def css(filepath):
    return static_file(filepath, root='./css')

@route('/js/<filepath:path>')
def js(filepath):
    return static_file(filepath, root='./js')

@route('/asset/<filepath:path>')
def asset(filepath):
    return static_file(filepath, root='./asset')

@route('/code', method='POST')
def do_code():
    method = request.forms.get('method')
    upload = request.files.get('upload-file')
    name, ext = os.path.splitext(upload.filename)
    if ext != '.csv':
        return 'File extension not allowed'


    df = read_csv(BytesIO(upload.file.read()))
    if method.lower() == 'statistics':
        summary = summary_data(df).to_json(orient='table')
        dataframe = df.to_json(orient='table')
        desc = data_shape(df)

        response = '{{"dataframe":{},"summary":{},"description":"{}"}}'.format(dataframe, summary, desc)

        return response
    elif method.lower() == 'classification':
        if check_independent_var(df) == 1:
            return '{"message":"All independent variables must be numeric"}'
        else:
            prediction = naive_bayes(df)
            accuracy = print_naive_bayes_accuracy(prediction)

            response = '{{"prediction":{},"accuracy":"{}"}}'.format(prediction.to_json(orient='table'), accuracy)
            return response
    elif method.lower() == 'clustering':
        k = request.forms.get('k')
        max_iteration = request.forms.get('max-iteration')
        
        if not k and not max_iteration:
            cluster, centroid = kmeans(df)
        elif not k:
            cluster, centroid = kmeans(df, max_iter=int(max_iteration))
        elif max_iteration == '':
            cluster, centroid = kmeans(df, k=int(k))
        else:
            cluster, centroid = kmeans(df, int(k), int(max_iteration))

        response = '{{"cluster":{},"centroid":{}}}'.format(cluster.to_json(orient='table'), centroid.to_json(orient='table'))
        return response

run(host='localhost', port=88, debug=True)