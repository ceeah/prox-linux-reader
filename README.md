This project aims to create alternative to U-Prox Desktop Reader, which
- Runs on Linux
- Utilizes cheap RFID readers, such as RDM6300.

Integrates well with U-Prox WEB on Linux.

## Reader protocol

- Frames follow the format described in `docs/RDM6300.txt` (`0x02 + 10 hex digits + 2 hex checksum + 0x03`).

## Install

- User must be in dialout group to use the reader.
- In Ubuntu install: pytohon3-aioserial