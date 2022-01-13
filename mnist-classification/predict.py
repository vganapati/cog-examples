# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import cog
from models import model_instance 
import imageio
import numpy as np

class Predictor(cog.Predictor):
    def setup(self):
      """Load the model into memory to make running multiple predictions efficient"""
      self.model, _, _ = model_instance(restore=True)

    @cog.input("input", type=cog.Path, help="Image to classify")
    def predict(self, input):
        """Run a single prediction on the model"""
        # Preprocess the image
        x = imageio.imread(input)
        if len(x.shape)==3:
            x = np.mean(x,axis=-1)
        x = np.resize(x, [28,28])
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)
        x = x/255.
        predictions = self.model(x, training=False)

        return(np.argmax(predictions))
