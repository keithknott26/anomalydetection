brew install postgresql pipx
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pipx install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
python3 main.py --log-dir=input_logs
