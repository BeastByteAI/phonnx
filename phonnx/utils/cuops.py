from os import getenv
from typing import Optional, List, Dict, Union, Any
import numpy as np
import json
import re


def extract_from_env(env_var_name: str) -> Optional[str]:
    if len(env_var_name) > 0 and env_var_name.startswith("env://"):
        return getenv(env_var_name[6:], None)


def extract_highest_priority(priority_str: str, env_var_name: str) -> Optional[str]:
    if len(priority_str) > 0:
        return priority_str
    return extract_from_env(env_var_name)


def create_nested_dict(keys: List[str], value: Any) -> Union[List, Dict]:
    next_key = keys[0]
    stripped_key = next_key.replace("[]", "")

    if len(keys) == 1:
        if "[]" in next_key:
            return {stripped_key: [value]}
        return {next_key: value}

    remaining_keys = keys[1:]

    if "[]" in next_key:
        return {stripped_key: [create_nested_dict(remaining_keys, value)]}
    return {next_key: create_nested_dict(remaining_keys, value)}


def create_batched_nested_dict(keys: List[str], values: List[Any]) -> Union[List, Dict]:
    assert "<<batch_dim>>" in keys, "keys must contain <<batch_dim>>"
    batch_index = keys.index("<<batch_dim>>")
    pre_batch_keys = keys[:batch_index]
    post_batch_keys = keys[batch_index + 1 :]

    if post_batch_keys:
        batched_list = [create_nested_dict(post_batch_keys, value) for value in values]
    else:
        batched_list = values

    if pre_batch_keys:
        outer_dict = create_nested_dict(pre_batch_keys, batched_list)
        return outer_dict
    else:
        return batched_list


def extract_value_from_nested_dict(response_dict: Dict, keys: List[str]) -> str:
    for key in keys:
        try:
            if key.isnumeric():
                key = int(key)
            response_dict = response_dict[key]
        except KeyError:
            try:
                key = str(key)
                response_dict = response_dict[key]
            except KeyError:
                return ""
    return response_dict


def batch_generator(lst: List, batch_size: int) -> List:
    if batch_size == 0:
        for i in lst:
            yield i
    else:
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]


def prepare_request_info(
    input_url: np.ndarray,
    input_key: np.ndarray,
    input_org: np.ndarray,
    url: str,
    default_url: str,
    key: str,
    org: str,
    header_key_auth: str,
    header_key_org: str,
    custom_header: str = "",
) -> Dict:
    final_url = extract_highest_priority(str(input_url.squeeze()), url) or default_url
    assert final_url is not None, "url must be provided"
    final_key = extract_highest_priority(str(input_key.squeeze()), key) or "-"
    final_org = extract_highest_priority(str(input_org.squeeze()), org) or "-"
    if len(custom_header) > 0:
        headers = json.loads(custom_header)
    else:
        headers = {"Content-Type": "application/json"}
    if final_key != "-":
        headers[header_key_auth] = final_key
    if final_org != "-":
        headers[header_key_org] = final_org
    return {"final_url": final_url, "headers": headers}

def extract_json_key_from_string(input_str, key) -> Optional[Any]:
    match = re.search(r'\{[^{}]*\}', input_str)
    if match:
        json_like_str = match.group(0)
        
        try:
            parsed_json = json.loads(json_like_str)
            extracted_value = parsed_json.get(key, None)
            return extracted_value
            
        except json.JSONDecodeError:
            try:
                # Replace single quotes with double quotes and try parsing again
                valid_json_str = json_like_str.replace("'", '"')
                parsed_json = json.loads(valid_json_str)
                extracted_value = parsed_json.get(key, None)
                return extracted_value
                
            except json.JSONDecodeError:
                pass
                
    return None