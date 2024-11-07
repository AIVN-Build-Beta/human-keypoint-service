

locust -f tests/loadtests/locustfile.py --headless -u 100 -r 10 --host http://localhost:8000 --run-time 1m --html=loadtest_report.html

