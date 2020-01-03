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
