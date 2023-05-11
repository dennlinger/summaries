"""
OpenAI-based generative models.
This packages the intermediate representations into a (natural-language) semin-structured prompt template,
and issues a request to the OpenAI API.
Note: Running these models requires your own OpenAI API key, and *will incur costs* depending on the model used.
"""
import os

import openai

from .GeneratorBase import Generator


class OpenAIGenerator(Generator):

    def __init__(self):
        super(OpenAIGenerator, self).__init__()

        pass
