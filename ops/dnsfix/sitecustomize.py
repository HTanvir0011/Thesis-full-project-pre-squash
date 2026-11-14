"""Loaded by the Kaggle CLI child processes through PYTHONPATH (set in ops/kg.py).

The home network's DNS resolver (83.147.255.216) sometimes answers www.kaggle.com / www.kaggleusercontent.com with 147.45.69.3, which is not Kaggle
(2026-11-01: every output download failed with CERTIFICATE_VERIFY_FAILED, correctly). These names are therefore resolved through 8.8.8.8.
Nothing else changes: the TLS certificate and host-name checks stay on, so a wrong address is still refused; any failure falls back to the system resolver.
Self-check: python ops/dnsfix/sitecustomize.py
"""
import random
import socket
import struct

_real = socket.getaddrinfo
HOSTS = {"www.kaggle.com", "www.kaggleusercontent.com", "api.kaggle.com"}


def _skip(r, i):  # index just past a (possibly compressed) name
    while r[i]:
        if r[i] & 0xC0 == 0xC0:
            return i + 2
        i += r[i] + 1
    return i + 1


def a_records(host, server="8.8.8.8"):
    tid = random.randrange(65536)
    q = struct.pack(">HHHHHH", tid, 0x0100, 1, 0, 0, 0) + b"".join(bytes([len(p)]) + p.encode() for p in host.split(".")) + b"\0" + struct.pack(">HH", 1, 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(3)
        s.sendto(q, (server, 53))
        r = s.recv(2048)
    rid, flags, qd, an = struct.unpack(">HHHH", r[:8])
    if rid != tid or flags & 15 or not an:  # wrong id, error code, or no answer
        return []
    i = 12
    for _ in range(qd):
        i = _skip(r, i) + 4
    ips = []
    for _ in range(an):
        i = _skip(r, i)
        typ, _, _, n = struct.unpack(">HHIH", r[i:i + 10])
        i += 10
        if typ == 1 and n == 4:
            ips.append(socket.inet_ntoa(r[i:i + 4]))
        i += n
    return ips


def getaddrinfo(host, *a, **k):
    if host in HOSTS:
        try:
            ips = a_records(host)
        except (OSError, struct.error, IndexError):  # timeout, UDP blocked, malformed reply
            ips = []
        if ips:
            return [x for ip in ips for x in _real(ip, *a, **k)]
    return _real(host, *a, **k)


socket.getaddrinfo = getaddrinfo

if __name__ == "__main__":
    for h in sorted(HOSTS):
        ips = a_records(h)
        print(h, ips)
        assert ips and all(x.count(".") == 3 for x in ips), f"no usable A record for {h}"
    assert "147.45.69.3" not in a_records("www.kaggle.com")
    print("self-check ok")
