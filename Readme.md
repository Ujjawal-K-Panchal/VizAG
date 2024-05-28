# VizAG: Vizion Augmented Generation

# Setup
1. Make & activate venv:
    ```
        python -m pip install virtualenv
        python -m venv venv4vizag
        source venv4vizag/bin/activate (Linux|Mac)
        venv4vizag\Scripts\activate (Windows) 
    ```
1. Install requirements: `python -m pip install -r requirements.txt` 

# Run tests.
1. Test Image Captioning: `python -m baseclip.py` (Image to String).
2. Test Retrieval: `python -m retrieval.py` (Search docs similar to user query.)
