#!/usr/bin/env python3
"""Expose 127.0.0.1:8888 (vLLM, host-network container) to external interfaces on the same port.
Zero-downtime: leaves the local vLLM binding untouched; forwards TCP to it."""
import socket, threading, sys

TARGET = ("127.0.0.1", 8888)
PORT = 8888
# External interfaces on this host: LAN (enP7s7, wlP9s9) + Tailscale
LISTEN_IPS = ["<LAN-IP-1>", "<LAN-IP-2>", "<Tailscale-IP>"]


def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(conn):
    try:
        up = socket.create_connection(TARGET, timeout=15)
    except OSError:
        conn.close()
        return
    t1 = threading.Thread(target=pipe, args=(conn, up), daemon=True)
    t2 = threading.Thread(target=pipe, args=(up, conn), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    conn.close()
    up.close()


def serve(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, PORT))
    s.listen(128)
    print(f"listening on {ip}:{PORT} -> {TARGET[0]}:{TARGET[1]}", flush=True)
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def main():
    started = 0
    for ip in LISTEN_IPS:
        try:
            threading.Thread(target=serve, args=(ip,), daemon=True).start()
            started += 1
        except OSError as e:
            print(f"bind {ip}:{PORT} failed: {e}", flush=True)
    if started == 0:
        sys.exit(1)
    while True:
        threading.Event().wait(3600)


if __name__ == "__main__":
    main()
