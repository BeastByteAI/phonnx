from onnxruntime_extensions import onnx_op, PyCustomOpDef
from phonnx.constants import AI_BEASTBYTE_PYOPS
from phonnx.utils.cuops import prepare_request_info
from phonnx.clients.rest import RESTClient
import json
from typing import List
import numpy as np

OP_NAME = "GPTChatCompletion"


def _prepare_messages(messages: np.ndarray, roles_list: List[str]):
    msg_list = []
    for message, role in zip(messages, roles_list):
        msg_list.append({"role": role, "content": message})
    return msg_list


@onnx_op(
    op_type=f"{AI_BEASTBYTE_PYOPS}::{OP_NAME}V1",
    inputs=[
        PyCustomOpDef.dt_string,  # messages
        PyCustomOpDef.dt_string,  # url
        PyCustomOpDef.dt_string,  # key
        PyCustomOpDef.dt_string,  # org
    ],
    outputs=[PyCustomOpDef.dt_string],
    attrs={
        "temperature": PyCustomOpDef.dt_float,
        "extra_params_json": PyCustomOpDef.dt_string,
        "model": PyCustomOpDef.dt_string,
        "key": PyCustomOpDef.dt_string,
        "org": PyCustomOpDef.dt_string,
        "url": PyCustomOpDef.dt_string,
        "default_url": PyCustomOpDef.dt_string,
        "roles": PyCustomOpDef.dt_string,
        "variant": PyCustomOpDef.dt_string,
    },
)
def gpt_chat_completion_v1(*s, **kwargs):
    assert s[1].size == 1, "url must be a scalar"
    assert s[2].size == 1, "key must be a scalar"
    assert s[3].size == 1, "org must be a scalar"
    temperature = kwargs.get("temperature", 0.7)
    extra_params_json = kwargs.get("extra_params_json", "")
    model = kwargs.get("model", "")
    key = kwargs.get("key", "")
    org = kwargs.get("org", "")
    url = kwargs.get("url", "")
    default_url = kwargs.get("default_url", "")
    roles = kwargs.get("roles", "")
    variant = kwargs.get("variant", "openai")
    return gpt_chat_completion_v1_impl(
        s,
        temperature,
        extra_params_json,
        model,
        key,
        org,
        url,
        default_url,
        roles,
        variant,
    )


# for testing simplicity, the implementation is another function
def gpt_chat_completion_v1_impl(
    s: List,
    temperature: float,
    extra_params_json: str,
    model: str,
    key: str,
    org: str,
    url: str,
    default_url: str,
    roles: str,
    variant: str = "openai",
) -> np.ndarray:
    messages = s[0]
    assert messages.ndim == 2, "messages must be a 2D tensor (batch_size, num_messages)"
    roles_list = roles.split(",")
    assert (
        len(roles_list) == messages.shape[1]
    ), "roles_list must match the number of messages"
    params = {
        "temperature": temperature,
    }
    assert variant in ["openai", "azure"], "variant must be one of [openai, azure]"
    if len(extra_params_json) > 0:
        params.update(json.loads(extra_params_json))
    if len(model) > 0:
        params["model"] = model
    info = prepare_request_info(
        s[1],
        s[2],
        s[3],
        url,
        default_url,
        key,
        org,
        "Authorization" if variant == "openai" else "api-key",
        "OpenAI-Organization",
    )
    final_url = info["final_url"]
    headers = info["headers"]
    if "Authorization" in headers.keys() and "Bearer" not in headers["Authorization"]:
        headers["Authorization"] = "Bearer " + headers["Authorization"]
    completions = []
    for message in messages:
        msg_list = _prepare_messages(message, roles_list)
        local_params = params.copy()
        local_params["messages"] = msg_list
        response = RESTClient.make_request(final_url, local_params, headers)
        try:
            completions.append(response["choices"][0]["message"]["content"])
        except KeyError:
            completions.append("")
    return np.array(completions, dtype=np.str_).reshape(-1, 1)
