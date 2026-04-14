from model import SentimentModel


def evaluate_models(data_path: str = "data/sample_reviews.csv") -> dict:
    """
    Thin wrapper that loads models and delegates evaluation to SentimentModel.evaluate().
    Returns a dict ready for direct consumption by app.py.
    """
    model_mgr = SentimentModel()
    model_mgr.load_models()
    return model_mgr.evaluate(data_path)
