# Modified for the anonymous main training and evaluation release.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public vLLM interfaces used for weight synchronization."""
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None

vllm_version = get_version("vllm")
if vllm_version is None or Version(vllm_version) < Version("0.7.0"):
    raise RuntimeError("The main implementation requires vLLM >= 0.7.0")
from vllm import LLM
from vllm.distributed import parallel_state

__all__ = ["LLM", "parallel_state", "vllm_version"]
