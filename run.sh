brew install postgresql
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt

python main.py --log-dir=input_logs
