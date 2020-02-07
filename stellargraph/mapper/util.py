# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading


class SeededSamplers:
    def __init__(self, create_with_seed, rs):
        self._create_with_seed = create_with_seed
        self._samplers = []
        self._lock = threading.Lock()
        self._rs = rs

    def __getitem__(self, index):
        self._lock.acquire()
        try:
            return self._samplers[index]
        except IndexError:
            # always create a new seeded sampler in ascending order of batch number
            # this ensures seeds are deterministic even when batches are run in parallel
            for n in range(len(self._samplers), index + 1):
                seed = self._rs.randint(0, 2 ** 32 - 1)
                self._samplers.append(self._create_with_seed(seed))
            return self._samplers[index]
        finally:
            self._lock.release()
