
# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================
from setfit import SetFitModel


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def question_classifier(question_body: str):
    print("Binary Classification In-Progess.....")
    model = SetFitModel.from_pretrained("sharukat/so_mpnet-base_question_classifier")
    prediction = model(question_body)
    print("Completed Successfully\n")
    return prediction