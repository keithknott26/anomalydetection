brew install postgresql pipx
git clone https://github.com/termcolor/termcolor
cd termcolor
python3 -m pip install .
cd ..
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
brew install pipx
pipx install -r requirements.txt
python3 -c "import nltk; nltk.download('stopwords')"
python3 main.py --log-dir=input_logs
