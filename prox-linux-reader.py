#!/usr/bin/env python3
"""WebSocket server that broadcasts RFID codes read from an RDM6300 (UART) reader."""

import argparse
import asyncio
import json
import logging
import sys
from typing import Set

try:
    import serial, aioserial
except ImportError:
    print("The 'aioserial' package is required. Install it with 'pip install aioserial'.")
    sys.exit(1)

try:
    from websockets.asyncio.server import ServerConnection, serve
except ImportError:  # pragma: no cover - dependency missing at runtime
    print("The 'websockets' package is required. Install it with 'pip install websockets'.")
    sys.exit(1)


DESCRIPTION = (
    "Linux alternative to U-Prox Desktop Reader that uses cheap RFID readers "
    "such as the RDM6300. Integrates with U-Prox WEB on Linux."
)

HOST = "localhost"
PORT = 40013
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600
PACKET_START = 0x02
PACKET_END = 0x03

CONNECTED_CLIENTS: Set[ServerConnection] = set()
SERIAL_EXCEPTIONS = serial.SerialException



class DebugPrefixFormatter(logging.Formatter):
    """Formatter that prefixes debug records so they are easy to spot."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - non-critical formatting
        message = super().format(record)
        if record.levelno == logging.DEBUG and not message.startswith("[D]"):
            return f"[D] {message}"
        return message


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(DebugPrefixFormatter("%(message)s"))
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.propagate = False

    noisy_loggers = ("asyncio", "websockets.server")
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


LOGGER = logging.getLogger("prox_linux_reader")


def parse_rdm6300_payload(raw_payload: str) -> str:
    """Validate RDM6300 payload and return 10 hex digits without checksum."""
    if not raw_payload:
        raise ValueError("Received empty payload, ignoring packet.")

    if len(raw_payload) != 12:
        raise ValueError(f"Unexpected payload length ({len(raw_payload)}): '{raw_payload}'. Ignoring packet.")

    card_hex = raw_payload[:10].upper()
    checksum_hex = raw_payload[10:]

    try:
        card_bytes = bytes.fromhex(card_hex)
        expected_checksum = int(checksum_hex, 16)
    except ValueError:
        raise ValueError(f"Malformed payload '{raw_payload}', ignoring packet.") from None

    computed_checksum = 0
    for byte in card_bytes:
        computed_checksum ^= byte

    if computed_checksum != expected_checksum:
        raise ValueError(
            f"Checksum mismatch for payload '{raw_payload}': "
            f"expected {expected_checksum:02X}, computed {computed_checksum:02X}. Ignoring packet."
        )

    return card_hex


async def register_client(websocket: ServerConnection) -> None:
    """Add a connected client to the in-memory registry."""
    CONNECTED_CLIENTS.add(websocket)
    peer = websocket.remote_address
    LOGGER.debug("Client connected: %s:%s", peer[0], peer[1])


async def unregister_client(websocket: ServerConnection) -> None:
    """Remove a client from the registry."""
    CONNECTED_CLIENTS.discard(websocket)
    peer = websocket.remote_address
    LOGGER.debug("Client disconnected: %s:%s", peer[0], peer[1])


async def ws_handler(websocket: ServerConnection) -> None:
    """Keep the connection alive until the client disconnects."""
    await register_client(websocket)
    try:
        await websocket.wait_closed()
    finally:
        await unregister_client(websocket)


async def broadcast(identifier: str) -> None:
    """Send the identifier payload to every connected client."""
    if not CONNECTED_CLIENTS:
        LOGGER.debug("No clients connected. Waiting for a connection...")
        return

    payload = json.dumps({"Identifiers": [identifier]})
    stale_clients = []
    sent_count = 0

    for client in list(CONNECTED_CLIENTS):
        try:
            await client.send(payload)
            sent_count += 1
        except Exception:
            stale_clients.append(client)

    for client in stale_clients:
        CONNECTED_CLIENTS.discard(client)

    LOGGER.debug("Sent payload '%s' to %s client(s).", identifier, sent_count)


async def serial_reader(queue: "asyncio.Queue[str]", serial_port: str) -> None:
    """Read RFID identifiers asynchronously using aioserial."""
    buffer = bytearray()
    collecting = False

    while True:
        serial_conn = None
        try:
            try:
                serial_conn = aioserial.AioSerial(port=serial_port, baudrate=BAUD_RATE, timeout=None)
                LOGGER.info("Reader connected on %s. Ready to accept cards.", serial_port)
            except SERIAL_EXCEPTIONS as exc:
                LOGGER.info("Waiting for reader...\r")
                LOGGER.debug(
                    "Unable to open %s (%s). Connect the reader and retrying in 1 second.",
                    serial_port,
                    exc,
                )
                await asyncio.sleep(1)
                continue

            buffer.clear()
            collecting = False

            while True:
                try:
                    byte = await serial_conn.read_async(1)
                except SERIAL_EXCEPTIONS as exc:
                    LOGGER.warning("Reader disconnected. Reconnecting...")
                    LOGGER.debug("Disconnect reason: %s", exc)
                    break

                if not byte:
                    await asyncio.sleep(0)
                    continue

                value = byte[0]
                if value == PACKET_START:
                    buffer.clear()
                    collecting = True
                elif value == PACKET_END and collecting:
                    collecting = False
                    if not buffer:
                        continue
                    try:
                        raw_payload = buffer.decode("ascii").strip()
                    except UnicodeDecodeError:
                        LOGGER.warning("Received non-ASCII data from reader, ignoring packet.")
                        buffer.clear()
                        continue

                    try:
                        card_hex = parse_rdm6300_payload(raw_payload)
                    except ValueError as exc:
                        LOGGER.warning("%s", exc)
                        buffer.clear()
                        continue

                    await queue.put(card_hex)
                    LOGGER.info("Read identifier: %s", card_hex)
                    buffer.clear()
                elif collecting:
                    buffer.append(value)
        finally:
            if serial_conn is not None:
                serial_conn.close()
                LOGGER.debug("Serial port closed.")

        await asyncio.sleep(1)


async def identifier_dispatcher(queue: "asyncio.Queue[str]") -> None:
    """Wait for identifiers from the queue and broadcast them."""
    while True:
        identifier = await queue.get()
        await broadcast(identifier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_SERIAL_PORT,
        help=f"Serial device for the reader (default: {DEFAULT_SERIAL_PORT})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


async def main(serial_port: str) -> None:
    identifier_queue: "asyncio.Queue[str]" = asyncio.Queue()
    async with serve(ws_handler, host=HOST, port=PORT):
        LOGGER.debug("WebSocket server running on ws://%s:%s", HOST, PORT)
        
        dispatcher_task = asyncio.create_task(identifier_dispatcher(identifier_queue))
        serial_task = asyncio.create_task(serial_reader(identifier_queue, serial_port))
        await asyncio.gather(serial_task, dispatcher_task)


if __name__ == "__main__":
    cli_args = parse_args()
    configure_logging(getattr(logging, cli_args.log_level))
    try:
        asyncio.run(main(cli_args.device))
    except KeyboardInterrupt:
        LOGGER.info("Shutting down server.")
