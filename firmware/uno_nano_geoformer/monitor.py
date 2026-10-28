#!/usr/bin/env python3
"""
Interactive Serial Terminal for GeoFormer Nano on Arduino Uno
Author: Shrikant Bhosale (Atmabhan Pandit)
"""

import serial
import sys
import time

PORT = "/dev/ttyACM0"
BAUD = 115200

def main():
    print(f"Connecting to GeoFormer Nano on {PORT} @ {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"Error opening {PORT}: {e}")
        sys.exit(1)

    time.sleep(1.8) # Wait for DTR reset
    print("Connected! Live interactive session started (Press Ctrl+C to exit):\n")

    # Read boot banner
    time.sleep(0.5)
    while ser.in_waiting:
        line = ser.read(ser.in_waiting).decode("utf-8", errors="replace")
        sys.stdout.write(line)
        sys.stdout.flush()

    try:
        while True:
            cmd = input("\nEnter command [1=2-Hop, 2=3-Hop, 3=Plasticity, 4=Lattice, 5=Benchmark, 6=Triage]: ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["q", "quit", "exit"]:
                break
            ser.write(cmd[0].encode("utf-8"))
            time.sleep(0.4)
            while ser.in_waiting:
                chunk = ser.read(ser.in_waiting).decode("utf-8", errors="replace")
                sys.stdout.write(chunk)
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nExiting interactive monitor.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
