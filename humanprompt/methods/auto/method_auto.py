from collections import OrderedDict

from ..ama_prompting.method import AMAPromptingMethod
from ..binder.method import BinderMethod
from ..cot.method import CoTMethod
from ..db_text2sql.method import DBText2SQLMethod
from ..react.method import ReActMethod
from ..standard.method import StandardMethod
from ..zero_shot_cot.method import ZeroShotCoTMethod
from .auto_factory import BaseAutoMethod
from ..batch_infererence.method import BatchInferenceMethod
from ..batch_infererence_chat.method import BatchInferenceChatMethod
from ..cot_chat.method import CoTChatMethod

METHOD_MAPPING_NAMES = OrderedDict(
    [
        # TODO: Method class definition should have different names for different methods
        ("standard", StandardMethod),
        ("cot", CoTMethod),
        ("zero_shot_cot", ZeroShotCoTMethod),
        ("react", ReActMethod),
        ("binder", BinderMethod),
        ("ama_prompting", AMAPromptingMethod),
        ("db_text2sql", DBText2SQLMethod),
        ("batch_inference", BatchInferenceMethod),
        ("cot_chat", CoTChatMethod),
        ("batch_inference_chat", BatchInferenceChatMethod),
    ]
)


class AutoMethod(BaseAutoMethod):
    _method_mapping = METHOD_MAPPING_NAMES
