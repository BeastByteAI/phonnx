from onnxruntime_extensions import onnx_op, PyCustomOpDef
from phonnx.constants import AI_BEASTBYTE_PYOPS
from phonnx.utils.cuops import (
    create_nested_dict,
    extract_value_from_nested_dict,
    prepare_request_info,
)
from phonnx.clients.rest import RESTClient
import json
from typing import List
import numpy as np

OP_NAME = "UniversalTextCompletion"


@onnx_op(
    op_type=f"{AI_BEASTBYTE_PYOPS}::{OP_NAME}V1",
    inputs=[
        PyCustomOpDef.dt_string,  # prompt
        PyCustomOpDef.dt_string,  # url
        PyCustomOpDef.dt_string,  # key
        PyCustomOpDef.dt_string,  # org
    ],
    outputs=[PyCustomOpDef.dt_string],
    attrs={
        "optional_params_json": PyCustomOpDef.dt_string,
        "key": PyCustomOpDef.dt_string,
        "org": PyCustomOpDef.dt_string,
        "header_key_auth": PyCustomOpDef.dt_string,
        "header_key_org": PyCustomOpDef.dt_string,
        "url": PyCustomOpDef.dt_string,
        "default_url": PyCustomOpDef.dt_string,
        "completion_location": PyCustomOpDef.dt_string,
        "prompt_location": PyCustomOpDef.dt_string,
        "custom_header": PyCustomOpDef.dt_string,
    },
)
def universal_completion_v1(*s, **kwargs):
    assert s[1].size == 1, "url must be a scalar"
    assert s[2].size == 1, "key must be a scalar"
    assert s[3].size == 1, "org must be a scalar"
    optional_params_json = kwargs.get("optional_params_json", "")
    key = kwargs.get("key", "")
    org = kwargs.get("org", "")
    url = kwargs.get("url", "")
    default_url = kwargs.get("default_url", "")
    header_key_auth = kwargs.get("header_key_auth", "")
    header_key_org = kwargs.get("header_key_org", "")
    prompt_location = kwargs.get("prompt_location", "prompt")
    completion_location = kwargs.get("completion_location", "0")
    custom_header = kwargs.get("custom_header", "")
    return universal_completion_v1_impl(
        s,
        optional_params_json,
        key,
        org,
        url,
        default_url,
        header_key_auth,
        header_key_org,
        prompt_location,
        completion_location,
        custom_header,
    )


def universal_completion_v1_impl(
    s: List,
    extra_params_json: str,
    key: str,
    org: str,
    url: str,
    default_url: str,
    header_key_auth: str,
    header_key_org: str,
    prompt_location: str,
    completion_location: str,
    custom_header: str,
) -> np.ndarray:
    prompts = s[0]
    assert prompts.ndim == 2, "prompts must be a 2D tensor (batch_size, 1)"
    assert prompts.shape[1] == 1, "prompts must be a 2D tensor (batch_size, 1)"
    params = {}
    if len(extra_params_json) > 0:
        params = json.loads(extra_params_json)
    info = prepare_request_info(
        s[1],
        s[2],
        s[3],
        url,
        default_url,
        key,
        org,
        header_key_auth,
        header_key_org,
        custom_header,
    )
    final_url = info["final_url"]
    headers = info["headers"]
    completions = []
    prompt_location_lst = prompt_location.split(",")
    completion_location_lst = completion_location.split(",")
    for prompt in prompts:
        request_body = create_nested_dict(prompt_location_lst, prompt[0])
        local_params = params.copy()
        if not isinstance(request_body, dict):
            request_body = {"inputs": request_body}
        request_body.update(local_params)
        response = RESTClient.make_request(final_url, request_body, headers)
        try:
            completions.append(
                str(extract_value_from_nested_dict(response, completion_location_lst))
            )
        except KeyError:
            completions.append("")
    return np.array(completions, dtype=np.str_).reshape(-1, 1)
