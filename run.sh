brew install postgresql
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
python3 main.py --log-dir=input_logs
