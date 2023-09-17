from onnxruntime_extensions import onnx_op, PyCustomOpDef
from phonnx.constants import AI_BEASTBYTE_PYOPS
from phonnx.utils.cuops import (
    create_nested_dict,
    extract_value_from_nested_dict,
    prepare_request_info,
    batch_generator,
    create_batched_nested_dict,
)
from phonnx.clients.rest import RESTClient
import json
from typing import List
import numpy as np

OP_NAME = "UniversalTextEmbedding"


@onnx_op(
    op_type=f"{AI_BEASTBYTE_PYOPS}::{OP_NAME}V1",
    inputs=[
        PyCustomOpDef.dt_string,  # prompt
        PyCustomOpDef.dt_string,  # url
        PyCustomOpDef.dt_string,  # key
        PyCustomOpDef.dt_string,  # org
    ],
    outputs=[PyCustomOpDef.dt_float],
    attrs={
        "optional_params_json": PyCustomOpDef.dt_string,
        "key": PyCustomOpDef.dt_string,
        "org": PyCustomOpDef.dt_string,
        "header_key_auth": PyCustomOpDef.dt_string,
        "header_key_org": PyCustomOpDef.dt_string,
        "url": PyCustomOpDef.dt_string,
        "default_url": PyCustomOpDef.dt_string,
        "embedding_location": PyCustomOpDef.dt_string,
        "prompt_location": PyCustomOpDef.dt_string,
        "custom_header": PyCustomOpDef.dt_string,
        "batch_size": PyCustomOpDef.dt_int64,
    },
)
def universal_embedding_v1(*s, **kwargs):
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
    embedding_location = kwargs.get("embedding_location", "0")
    custom_header = kwargs.get("custom_header", "")
    batch_size = int(kwargs.get("batch_size", 0))
    return universal_embedding_v1_impl(
        s,
        optional_params_json,
        key,
        org,
        url,
        default_url,
        header_key_auth,
        header_key_org,
        prompt_location,
        embedding_location,
        custom_header,
        batch_size,
    )


# for testing simplicity, the implementation is another function
def universal_embedding_v1_impl(
    s: List,
    extra_params_json: str,
    key: str,
    org: str,
    url: str,
    default_url: str,
    header_key_auth: str,
    header_key_org: str,
    prompt_location: str,
    embedding_location: str,
    custom_header: str,
    batch_size: int,
) -> np.ndarray:
    assert s[0].ndim == 2, "prompts must be a 2D tensor (batch_size, 1)"
    assert s[0].shape[1] == 1, "prompts must be a 2D tensor (batch_size, 1)"
    prompts = s[0].squeeze(1)
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
    embedding_location_lst = embedding_location.split(",")
    if batch_size == 0 and "<<batch_dim>>" in prompt_location_lst:
        raise ValueError(
            "<<batch_dim>> is used in prompt_location but batch_size is 0."
        )
    if batch_size > 0:
        assert (
            "<<batch_dim>>" in prompt_location_lst
        ), "<<batch_dim>> must be in prompt_location"
        assert (
            "<<batch_dim>>" in embedding_location_lst
        ), "<<batch_dim>> must be in embedding_location"
        batch_dim_location_out = embedding_location_lst.index("<<batch_dim>>")
    create_dict_fn = (
        create_nested_dict if batch_size == 0 else create_batched_nested_dict
    )
    for prompt in batch_generator(prompts, batch_size):
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        request_body = create_dict_fn(prompt_location_lst, prompt)
        local_params = params.copy()
        if not isinstance(request_body, dict):
            request_body = {"inputs": request_body}
        request_body.update(local_params)
        response = RESTClient.make_request(final_url, request_body, headers)
        if batch_size == 0:
            embedding = extract_value_from_nested_dict(response, embedding_location_lst)
            completions.append(embedding)
        else:
            batch_embeddings = []
            for i in range(batch_size):
                try:
                    embedding_location_lst[batch_dim_location_out] = str(i)
                    embedding = extract_value_from_nested_dict(
                        response, embedding_location_lst
                    )
                    batch_embeddings.append(embedding)
                except IndexError:
                    break
            completions.extend(batch_embeddings)
    if len(completions) != len(prompts):
        raise RuntimeError("Number of embeddings did not match number of prompts")
    return np.asarray(completions, dtype=np.float32)