## Introduction
This project aims to create alternative to U-Prox Desktop Reader to use cheap readers with U-Prox WEB.

Key features: 
- Runs on both Linux and Windows
- Utilizes cheap RFID readers, such as RDM6300 (can be extended in the future)

## Reader protocol

- Frames follow the format described in `docs/RDM6300.txt` (`0x02 + 10 hex digits + 2 hex checksum + 0x03`).

## Install

- User must be in `dialout` group to use the reader.
- In Ubuntu install: `apt install pytohon3-aioserial python3-websockets`

## Usage
```
usage: prox-linux-reader.py [-h] [-D DEVICE] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Linux alternative to U-Prox Desktop Reader that uses cheap RFID readers such as the RDM6300. Integrates with U-Prox WEB on Linux.

options:
  -h, --help            show this help message and exit
  -D, --device DEVICE   Serial device for the reader (default: /dev/ttyUSB0)
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO).
```

## Про проект
Ця програма дозволяє використовувати дешевші зчитувачі (наприклад, RDM6300) як альтернативу U-Prox Desktop Reader на Linux і Windows для роботи з U-Prox WEB
