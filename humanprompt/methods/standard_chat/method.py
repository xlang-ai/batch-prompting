from typing import Any, Dict, List, Optional, Union

import openai

from ...components.prompt import PromptChatBuilder
from ..base_method.method import PromptChatMethod
from ...components.post_hoc import HocPoster


class StandardChatMethod(PromptChatMethod):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def run(
        self,
        x: Union[str, Dict],
        prompt_file_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, List[str]]:
        verbose = kwargs.get("verbose", False)
        messages = PromptChatBuilder.build_chat_prompt(
            x=x,
            prompt_file_path=prompt_file_path,
            transform=kwargs["transform"]
            if "transform" in kwargs
            else self.kwargs.get("transform", None),
        )

        if verbose:
            print(f"Message::\n{messages}")

        openai_chat_input_params = self._build_chat_input_params(self.kwargs)
        response = openai.ChatCompletion.create(
            messages=messages,
            **openai_chat_input_params
        )

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
