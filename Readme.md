# VizAG: Vision Augmented Generation
<a href = "https://drive.google.com/file/d/1FuJIFqK3hxGMVM90KvImHjLJkihvI4TM/view?usp=sharing">![Untitled presentation-2](https://github.com/Ujjawal-K-Panchal/VizAG/assets/31011628/94ed31da-3932-422d-bd2c-4e4f55d9984b)
</a>

# Setup
1. Make & activate venv:
    ```
        python -m pip install virtualenv
        python -m venv venv4vizag
        source venv4vizag/bin/activate (Linux|Mac)
        venv4vizag\Scripts\activate (Windows) 
    ```
2. Install requirements: `python -m pip install -r requirements.txt` 
3. Make a `secrets/` directory; ask ujjawal for `hf_token` file and put inside it.

# Run tests.
1. Test Image Captioning: `python baseclip.py` (Image to String).
2. Test Retrieval: `python retrieval.py` (Search docs similar to user query.)
3. Test Generation: `python generation.py` (Test Generation; note need `~6 GB GPU VRAM`).
4. Test ViZAG: `python vizag.py`
