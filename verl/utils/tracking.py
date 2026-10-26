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
"""Console logging for anonymous experiment runs."""
from verl.utils.logger.aggregate_logger import LocalLogger

class Tracking:
    def __init__(self, project_name, experiment_name, default_backend="console", config=None):
        backends = [default_backend] if isinstance(default_backend, str) else list(default_backend)
        if backends != ["console"]:
            raise ValueError("The submission package supports console logging")
        self.logger = LocalLogger(print_to_console=True)
        self._finished = False
        self._finish_exit_code = 0

    def log(self, data, step, backend=None):
        self.logger.log(data=data, step=step)

    def finish(self, exit_code=0):
        self._finish_exit_code = exit_code
        self._finished = True

class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        # Validation metrics are logged through Tracking; optional sample tables are disabled.
        return None
