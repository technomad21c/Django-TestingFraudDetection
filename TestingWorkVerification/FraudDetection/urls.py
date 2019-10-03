from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from .views import TestView, index, setup_fraud_detection, verify_testing_works

urlpatterns = [
    path('test/<str:name>/', index, name='index'),
    path('ml/setup/', setup_fraud_detection, name='fraud_detection_setup'),
    path('ml/verify/', verify_testing_works, name='fraud_verification'),
    path('class/<str:name>/', csrf_exempt(TestView.as_view()), name='test_class'),
    # path('mine/', MyView.as_view(), name='my-view'),
]