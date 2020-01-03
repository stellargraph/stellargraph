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
#! /usr/bin/env python
import hashlib, hmac, os, stat, sys
## Return the hash of the contents of the specified file, as a hex string
def file_hash(name):
    h = hashlib.sha256()
    with open(name, 'rb') as f:
        while True:
            buf = f.read(16384)
            if len(buf) == 0: break
            h.update(buf)
    return h

## Traverse the specified path and update the hash with a description of its
## name and contents
def traverse(path):
    rs = os.lstat(path)
    quoted_name = repr(path)
    if stat.S_ISDIR(rs.st_mode):
        h = hashlib.sha256()
        h.update(f'dir {quoted_name}\n'.encode("utf-8"))
        for entry in sorted(os.listdir(path)):
            h.update(traverse(os.path.join(path, entry)))
    elif stat.S_ISREG(rs.st_mode):
        h = file_hash(path)
    else: return b"" # silently symlinks and other special files

    print(f"{h.hexdigest()} {path}")
    return h.digest()

for root in sys.argv[1:]: traverse(root )
