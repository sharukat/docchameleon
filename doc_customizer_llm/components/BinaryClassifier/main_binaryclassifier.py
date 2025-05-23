from setfit import SetFitModel


def question_classifier(question_body: str):
    model = SetFitModel.from_pretrained(
        "sharukat/so_mpnet-base_question_classifier")
    prediction = model(question_body)
    return prediction

