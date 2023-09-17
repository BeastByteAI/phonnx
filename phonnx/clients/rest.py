import urllib
import json
import time
from typing import Dict, Optional, List


class RESTClient:
    max_retries = 3
    max_wait_between_retries = 60
    timeout = 600
    whitelist: Optional[List[str]] = None

    @classmethod
    def make_request(
        cls,
        url: str,
        payload: Dict,
        headers: Optional[Dict] = None,
    ) -> Dict:
        error_messages = []
        print(url, payload, headers)
        cls._check_whitelist(url)
        for i in range(cls.max_retries):
            try:
                req = urllib.request.Request(
                    url=url,
                    headers=headers if headers else {},
                    data=json.dumps(payload).encode("utf-8"),
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=cls.timeout) as response:
                    if response.status == 200:
                        return json.load(response)
                    else:
                        error_message = f"Attempt {i+1}: Failed with status code {response.status}: {response.read().decode('utf-8')}"
                        error_messages.append(error_message)
                        time.sleep(min(2**i, cls.max_wait_between_retries))

            except Exception as e:
                error_message = f"Attempt {i+1}: An exception occurred: {e}"
                error_messages.append(error_message)
                time.sleep(min(2**i, cls.max_wait_between_retries))

        joined_error_messages = "\n".join(error_messages)
        raise Exception(
            f"Failed to make a request after {cls.max_retries} retries. Errors:\n{joined_error_messages}"
        )

    @classmethod
    def _check_whitelist(cls, url) -> bool:
        domain = urllib.parse.urlparse(url).netloc
        if cls.whitelist and domain not in cls.whitelist:
            raise Exception(f"The domain {domain} is not in the whitelist.")
