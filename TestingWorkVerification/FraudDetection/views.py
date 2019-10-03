import os
import pickle
import re

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from rest_framework import status
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_swagger import renderers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


@api_view(['GET'])
def index(request, name):
    return HttpResponse("Hey " + name + ", it works well!")


@api_view(['POST'])
# @renderer_classes([renderers.OpenAPIRenderer, renderers.SwaggerUIRenderer])
def setup_fraud_detection(request):
    # save log file and api file from client
    bounty_id = request.POST.get('bountyId')
    logfile = bounty_id + '_' + request.POST.get('logfilename')
    apifile = bounty_id + '_' + request.POST.get('apifilename')
    files_dir = settings.TESTING_FILES_ROOT
    save_path = os.path.join(files_dir, '', logfile)
    path = default_storage.save(save_path, request.FILES['logfile'])
    save_path = os.path.join(files_dir, '', apifile)
    path = default_storage.save(save_path, request.FILES['apifile'])

    # generate a machine learning model to detect fraud on testing works
    generate_model(bounty_id, os.path.join(files_dir, logfile), os.path.join(files_dir, apifile))

    return Response(default_storage.path(path), status.HTTP_200_OK)


@api_view(['POST'])
def verify_testing_works(request):
    bounty_id = request.POST.get('bountyId')
    logfile = bounty_id + '_TestingResult_' + request.POST.get('logfilename')
    apifile = bounty_id + '_restapi.csv'
    files_dir = settings.TESTING_FILES_ROOT
    save_path = os.path.join(files_dir, '', logfile)
    path = default_storage.save(save_path, request.FILES['logfile'])

    clf, enc = load_model(bounty_id)
    feature_log = retrieve_feature(os.path.join(files_dir, logfile))
    result = predict(clf, enc, os.path.join(files_dir, logfile))

    return Response(result, status.HTTP_200_OK)


def generate_model(bounty_id, logfile, apifile):
    feature_data = retrieve_feature(logfile)
    api_pattern = retrieve_api(apifile)
    labeled_data = label_data(bounty_id, feature_data, api_pattern)

    feature_data = retrieve_feature(logfile)
    pattern_api = retrieve_api(apifile)
    label_data_file = label_data(bounty_id, feature_data, pattern_api)
    methods, urls = distinct_field(distinct_data(label_data_file, feature_data))
    detection_model, enc = train_model(methods, urls, label_data_file)
    store_model(bounty_id, detection_model, enc)


def retrieve_feature(logfile):
    data = []
    with open(logfile, 'r') as logfile:
        for line in logfile:
            if ("Mapped \"{[" in line):
                temp_url = re.search(r'\{\[(.*?)\]', line)
                base_url = re.sub("\{.*?\}", "", temp_url.group(1)).rstrip('/')
                patterns = ['/swagger-resources', '/error']
                if any(pattern in base_url for pattern in patterns):
                    continue
                temp_method = re.search(r'methods=\[(.*?)\]', line)
                if temp_method:
                    method = temp_method.group(1)
                else:
                    method = "NULL"
                    #                 data_line = "Base Url: {0:<50s} Methods: {1}".format(base_url+",", method)
                data_row = method.lower() + "," + base_url
                data.append(data_row)

    return data


def retrieve_api(apifile):
    patterns = []
    with open(apifile, 'r') as file:
        for line in file:
            method, url = line.split(",")
            patterns.append(method.lower() + "," + url.rstrip('\n'))

    return patterns


def label_data(bounty_id, feature_data, patterns):
    label_data_file = bounty_id + ".lbl"
    with open(label_data_file, 'w') as file:
        for d in feature_data:
            handson_test = '0'
            if any(pattern in d.replace(" ", "") for pattern in patterns):
                handson_test = '1'
            label_data = d + ',' + handson_test + '\n'
            file.write(label_data)

    return label_data_file

def distinct_data(label_data_file, feature_data_file):
    distinct_data = []
    with open(label_data_file, 'r') as featurefile:
        for d in featurefile:
            if isNewData(d, distinct_data):
                distinct_data.append(d)

    return distinct_data


def isNewData(d, data):
    if any(x in d for x in data):
        return False
    else:
        return True

def distinct_field(distinct_data):
    distinct_method = []
    distinct_url = []
    distinct_label = []
    feature_data = []
    for f in distinct_data:
        m, u, l = f.rstrip('\n').split(',')
        if (isNewData(m, distinct_method)):
            distinct_method.append(m)
        if (isNewData(u, distinct_url)):
            distinct_url.append(u)
        if (isNewData(l, distinct_label)):
            distinct_label.append(l)
        feature_data.append(m + ',' + u)

    return distinct_method, distinct_url


def train_model(distinct_method, distinct_url, label_data_file):
    enc = preprocessing.OneHotEncoder(categories=[distinct_method, distinct_url], handle_unknown='ignore')
    col_names = ['method', 'url', 'label']
    feature_cols = ['method', 'url']
    apitest = pd.read_csv(label_data_file, header=None, names=col_names)
    # apitest.label = apitest.label.astype(str)
    X = apitest[['method', 'url']].values.tolist()
    y = apitest.label

    enc.fit(X)
    X = enc.transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    return clf, enc


def store_model(bounty_id, clf, enc):
    clf_serial = pickle.dumps(clf)
    enc_serial = pickle.dumps(enc)
    # save a serialized classifier in mongodb
    clf_serial = pickle.dumps(clf)
    client = MongoClient('localhost:27017')
    db = client.Axius_FraudDetection
    db.classifier.insert_one({'bountyId': bounty_id, 'model': clf_serial, 'enc': enc_serial})


def load_model(bounty_id):
    client = MongoClient('localhost:27017')
    db = client.Axius_FraudDetection
    data = db.classifier.find_one({'bountyId': bounty_id})
    # loaded_clf = pickle.load(open(clf_file, 'rb'))
    clf_serial = data['model']
    clf = pickle.loads(clf_serial)
    enc_serial = data['enc']
    enc = pickle.loads(enc_serial)

    return clf, enc


def predict(clf, enc, input_data_file):
    feature_data = retrieve_feature(input_data_file)
    input_data = convert_feature(feature_data)
    input = enc.transform(input_data).toarray()
    result = clf.predict(input).tolist()
    num_real_testing_trial = result.count(1)
    percent_real_testing_trial = num_real_testing_trial / len(result) * 100

    response = { 'trials': num_real_testing_trial, 'percent': percent_real_testing_trial}
    return response


def convert_feature(feature_data):
    data = []
    for d in feature_data:
        method, url = d.split(',')
        data.append([method, url])

    return data


class TestView(APIView):
    permission_classes = [AllowAny]
    # renderer_classes = [
    #     renderers.OpenAPIRenderer,
    #     renderers.SwaggerUIRenderer
    # ]
    def get(self, request, name):
        return HttpResponse("Hello " + name + ",  this is a message from GET of Class View")

    def post(self, request, name):
        return HttpResponse("Hello " + name + ", this is a message from POST of Class View")
