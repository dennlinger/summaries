"""
OpenAI-based generative models.
This packages the intermediate representations into a (natural-language) semin-structured prompt template,
and issues a request to the OpenAI API.
Note: Running these models requires your own OpenAI API key, and *will incur costs* depending on the model used.
"""
import os
import openai

from typing import Optional

from .GeneratorBase import Generator
from ..document import Segment

default_prompt = "Pretend to be a human tasked with re-writing individual sentences into a coherent text. " \
                 "Given the following intermediate sentences, re-write individual sentences into a cohesive segment, " \
                 "optionally reordering content. Newlines indicate a new sentence." \
                 "Sentence are ranked in decending order by their perceived importance.\n"


class OpenAIGenerator(Generator):
    OPENAI_API_KEY: str

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super(OpenAIGenerator, self).__init__()
        if api_key is not None:
            self.OPENAI_API_KEY = api_key
        else:
            self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            if self.OPENAI_API_KEY is None:
                raise ValueError("Either manually specify an API key for the OpenAI model, "
                                 "or alternatively set OPENAI_API_KEY as an environment variable.")

    def generate(self,
                 segments: list[Segment],
                 ex_ante_aspects: dict,
                 ex_post_aspects: dict,
                 custom_prompt: Optional[str] = None):
        # TODO: Consider moving to langchain for easier access?

        if not custom_prompt:
            custom_prompt = default_prompt

        segment_text = "\n".join([segment.raw for segment in segments])
        custom_prompt += "Intermediate sentences:\n"
        custom_prompt += segment_text
        custom_prompt += "\nAnswer:"

        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=[{"role": "user", "content": custom_prompt}])
        return completion.choices[0].message.content
