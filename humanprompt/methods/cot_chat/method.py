from typing import Any, Dict, List, Optional, Union

import json
import openai

from ...components.post_hoc import HocPoster
from ...components.prompt import PromptChatBuilder
from ..base_method.method import PromptChatMethod


class CoTChatMethod(PromptChatMethod):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def run(
            self,
            x: Union[str, Dict],
            **kwargs: Any
    ) -> Union[str, List[str]]:
        verbose = kwargs.get("verbose", False)
        messages = PromptChatBuilder.build_chat_prompt(
            x=x,
            prompt_file_path=self.kwargs["prompt_file_path"],
            transform=kwargs["transform"]
            if "transform" in kwargs
            else self.kwargs.get("transform", None),
        )

        if verbose:
            print(f"Message::\n{json.dumps(messages[-1], indent=2)}")

        openai_chat_input_params = self._build_chat_input_params(self.kwargs)
        max_retries, num_retries = 5, 0
        while num_retries < max_retries:
            num_retries += 1
            try:
                response = openai.ChatCompletion.create(
                    messages=messages,
                    **openai_chat_input_params
                )
                response = response["choices"][0]["message"]["content"]
                response = response.replace("\n", " ")
                break
            except Exception as e:
                print(f"Error::\n{e}, retry times: {num_retries}")
                response = "OpenAI API is currently down."

        if verbose:
            print(f"Response::\n{response}")

        y = HocPoster.post_hoc(
            response,
            extract=kwargs["extract"]
            if "extract" in kwargs
            else self.kwargs.get("extract", None),
            aggregation=kwargs["aggregation"]
            if "aggregation" in kwargs
            else self.kwargs.get("aggregation", None),
            extraction_regex=kwargs["extraction_regex"]
            if "extraction_regex" in kwargs
            else self.kwargs.get("extraction_regex", None),
        )
        return y

    def _build_chat_input_params(self, config: Dict) -> Dict:
        input_params = {
            "model": config.get("engine", None),
            "max_tokens": config.get("max_tokens", None),
            "temperature": config.get("temperature", None),
            "stop": config.get("stop_sequence", None),
        }
        return input_params