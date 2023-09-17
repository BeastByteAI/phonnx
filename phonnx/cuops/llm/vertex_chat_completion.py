from onnxruntime_extensions import onnx_op, PyCustomOpDef
from phonnx.constants import AI_BEASTBYTE_PYOPS
from phonnx.utils.cuops import prepare_request_info
from phonnx.clients.rest import RESTClient
from typing import List, Dict
import numpy as np

OP_NAME = "VertexChatCompletion"


def _prepare_messages(messages: np.ndarray, roles_list: List[str]) -> Dict:
    context = None
    examples_in = []
    examples_out = []
    examples = []
    messages_list = []
    known_roles = ("user", "bot")

    for role, message in zip(roles_list, messages):
        if role in known_roles:
            messages_list.append({"author": role, "content": message})
        elif role == "context":
            context = message
        elif role == "example_in":
            examples_in.append(message)
        elif role == "example_out":
            examples_out.append(message)
        else:
            raise ValueError(f"Unknown role {role}")
    assert len(examples_in) == len(
        examples_out
    ), "Number of examples_in and examples_out must match"
    request_body = {
        "messages": messages_list,
    }
    if len(examples_in) > 0:
        for example_in, example_out in zip(examples_in, examples_out):
            examples.append(
                {"input": {"content": example_in}, "output": {"content": example_out}}
            )
        request_body["examples"] = examples
    if context is not None:
        request_body["context"] = context
    return {"instances": [request_body]}


@onnx_op(
    op_type=f"{AI_BEASTBYTE_PYOPS}::{OP_NAME}V1",
    inputs=[
        PyCustomOpDef.dt_string,  # messages
        PyCustomOpDef.dt_string,  # url
        PyCustomOpDef.dt_string,  # key
    ],
    outputs=[PyCustomOpDef.dt_string],
    attrs={
        "temperature": PyCustomOpDef.dt_float,
        "maxOutputTokens": PyCustomOpDef.dt_int64,
        "topP": PyCustomOpDef.dt_float,
        "topK": PyCustomOpDef.dt_int64,
        "key": PyCustomOpDef.dt_string,
        "url": PyCustomOpDef.dt_string,
        "default_url": PyCustomOpDef.dt_string,
        "roles": PyCustomOpDef.dt_string,
    },
)
def vertex_chat_completion_v1(*s, **kwargs):
    assert s[1].size == 1, "url must be a scalar"
    assert s[2].size == 1, "key must be a scalar"
    key = kwargs.get("key", "")
    url = kwargs.get("url", "")
    default_url = kwargs.get("default_url", "")
    roles = kwargs.get("roles", "")
    maxOutputTokens = kwargs.get("maxOutputTokens", 4096)
    temperature = kwargs.get("temperature", 0.7)
    topP = kwargs.get("topP", 0.95)
    topK = kwargs.get("topK", 40)

    return vertex_chat_completion_v1_impl(
        s,
        temperature,
        maxOutputTokens,
        topP,
        topK,
        key,
        url,
        default_url,
        roles,
    )


# for testing simplicity, the implementation is another function
def vertex_chat_completion_v1_impl(
    s: List,
    temperature: float,
    maxOutputTokens: int,
    topP: float,
    topK: int,
    key: str,
    url: str,
    default_url: str,
    roles: str,
) -> np.ndarray:
    messages = s[0]
    assert messages.ndim == 2, "messages must be a 2D tensor (batch_size, num_messages)"
    roles_list = roles.split(",")
    assert (
        len(roles_list) == messages.shape[1]
    ), "roles_list must match the number of messages"
    params = {
        "temperature": temperature,
        "maxOutputTokens": maxOutputTokens,
        "topP": topP,
        "topK": topK,
    }
    info = prepare_request_info(
        s[1],
        s[2],
        np.asarray(""),
        url,
        default_url,
        key,
        "",
        "Authorization",
        "",
    )
    final_url = info["final_url"]
    headers = info["headers"]
    if "Authorization" in headers.keys() and "Bearer" not in headers["Authorization"]:
        headers["Authorization"] = "Bearer " + headers["Authorization"]
    completions = []
    for message in messages:
        msg_list = _prepare_messages(message, roles_list)
        request_body = {"parameters": params}
        request_body.update(msg_list)
        response = RESTClient.make_request(final_url, request_body, headers)
        try:
            completions.append(response["predictions"][0]["candidates"][0]["content"])
        except KeyError:
            completions.append("")
    return np.array(completions, dtype=np.str_).reshape(-1, 1)
