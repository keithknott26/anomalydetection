brew install postgresql pipx
git clone https://github.com/termcolor/termcolor
cd termcolor
python3 -m pip install .
cd ..
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
brew install pipx
pip3 install nltk numpy scikit-learn joblib prettytable python-Levenshtein APScheduler psycopg2-binary drain3 prettytable Levenshtein termcolor rich dash loguru joblib python-dateutil
pipx install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
python3 main.py --log-dir=input_logs
