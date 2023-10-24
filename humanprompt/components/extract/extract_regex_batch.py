import re
from typing import Any

from .extract_base import Extract


class QABatchExtract(Extract):
    @staticmethod
    def extract(raw_response: str, **kwargs: Any) -> str:
        """
        Extract the batch of QA answers from raw_response by regex.

        Args:
            raw_response: raw response from model
            **kwargs: other arguments
        Returns: extracted result list

        """
        batch_answers = []
        answer_idx = 0
        for answer in raw_response.split("\n"):
            # Skip answer prefix # TODO: Handle this in a better way
            if answer.startswith("A:"):
                answer = answer[len("A: "):]
                if len(batch_answers) + 1 < answer_idx:  # The gap > 1 means skip one answer, so we fill it with <empty>
                    batch_answers.append("<empty>")
                answer_idx += 1
            elif answer.startswith("A["):
                answer = answer[len("A[i]: "):]
                if len(batch_answers) + 1 < answer_idx:
                    batch_answers.append("<empty>")
                answer_idx += 1
            elif answer.startswith("Answer: "):
                answer = answer[len("Answer: "):]
                if len(batch_answers) + 1 < answer_idx:
                    batch_answers.append("<empty>")
                answer_idx += 1
            elif answer.startswith("Answer["):
                answer = answer[len("Answer[i]: "):]
                if len(batch_answers) + 1 < answer_idx:
                    batch_answers.append("<empty>")
                answer_idx += 1

            if "extraction_regex" in kwargs \
                    and kwargs["extraction_regex"] is not None:
                # if extraction_words is specified, we use it to extract the answer
                extraction_regex = kwargs["extraction_regex"]
                answer = re.match(extraction_regex, answer)
                if answer is None:
                    answer = "<empty>"
                else:
                    answer = answer.group(1)

            if answer != "<empty>":
                batch_answers.append(answer.lower())

        if len(batch_answers) < answer_idx:
            batch_answers.append("<empty>")

        return batch_answers
