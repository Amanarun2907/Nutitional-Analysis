import unittest
from src.models.train import train_model
from src.models.predict import make_prediction
import pandas as pd

class TestNutritionalAnalysisModels(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'calories': [100, 200, 150],
            'protein_g': [10, 20, 15],
            'fat_total_g': [5, 10, 7],
            'carbs_g': [15, 30, 20]
        })
        self.model = train_model(self.sample_data)

    def test_model_training(self):
        self.assertIsNotNone(self.model, "Model should be trained and not None")

    def test_prediction(self):
        sample_input = pd.DataFrame({
            'calories': [120],
            'protein_g': [12],
            'fat_total_g': [6],
            'carbs_g': [18]
        })
        prediction = make_prediction(self.model, sample_input)
        self.assertIsInstance(prediction, (int, float), "Prediction should be a numeric value")

if __name__ == '__main__':
    unittest.main()