from main_scaper import ai_webscraper

# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================

urls=[
        "https://www.geeksforgeeks.org/save-and-load-models-in-tensorflow/",
        "https://www.tensorflow.org/tutorials/keras/save_and_load",
        "https://www.tensorflow.org/guide/saved_model"
    ]
ai_webscraper(web_page="https://www.geeksforgeeks.org/save-and-load-models-in-tensorflow/", response_type="code")