#!/usr/bin/env python3
"""U-Prox Desktop like service that broadcasts RFID codes read from supported UART readers (F02DC, RDM6300)."""

import argparse
import asyncio
import json
import logging
import sys
from typing import Callable, Set

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
    "U-Prox Desktop like reader service that uses cheap RFID readers "
    "Integrates with U-Prox WEB and U-Prox desktop client"
)

HOST = "localhost"
PORT = 40013
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_READER = "F02DC"
BAUD_RATE = 9600
PACKET_START = 0x02
PACKET_END = 0x03
IDENTIFIER_COOLDOWN_SECONDS = 1.0

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
ParsedResult = tuple[str, bytes]


class UProxWebSocketServer:
    """Manages websocket clients and broadcasting identifiers."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.connected: Set[ServerConnection] = set()

    async def register(self, websocket: ServerConnection) -> None:
        self.connected.add(websocket)
        peer = websocket.remote_address
        LOGGER.info("U-Prox Client connected: %s:%s", peer[0], peer[1])

    async def unregister(self, websocket: ServerConnection) -> None:
        self.connected.discard(websocket)
        peer = websocket.remote_address
        LOGGER.debug("U-Prox Client disconnected: %s:%s", peer[0], peer[1])

    async def handler(self, websocket: ServerConnection) -> None:
        await self.register(websocket)
        try:
            await websocket.wait_closed()
        finally:
            await self.unregister(websocket)

    async def broadcast(self, identifier: str) -> None:
        if not self.connected:
            LOGGER.debug("No clients connected. Waiting for a connection...")
            return

        payload = json.dumps({"Identifiers": [identifier]})
        stale_clients = []
        sent_count = 0

        for client in list(self.connected):
            try:
                await client.send(payload)
                sent_count += 1
            except Exception:
                stale_clients.append(client)

        for client in stale_clients:
            self.connected.discard(client)

        LOGGER.debug("Sent payload '%s' to %s client(s).", identifier, sent_count)


class RDM6300PacketReader:
    """Stateful parser for RDM6300 frames."""

    name = "RDM6300"

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.collecting = False

    def reset(self) -> None:
        self.buffer.clear()
        self.collecting = False

    @staticmethod
    def _parse_frame(raw_payload: str) -> str:
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

    def feed(self, byte_value: int) -> ParsedResult | None:
        if byte_value == PACKET_START:
            self.buffer.clear()
            self.collecting = True
            return None

        if byte_value == PACKET_END and self.collecting:
            self.collecting = False
            if not self.buffer:
                return None
            try:
                raw_payload = self.buffer.decode("ascii").strip()
            except UnicodeDecodeError:
                raise ValueError("Received non-ASCII data from RDM6300 reader, ignoring packet.")

            card_hex = self._parse_frame(raw_payload)
            self.buffer.clear()
            frame_bytes = bytes([PACKET_START]) + raw_payload.encode("ascii") + bytes([PACKET_END])
            return card_hex, frame_bytes

        if self.collecting:
            self.buffer.append(byte_value)
        return None


class F02DCPacketReader:
    """Stateful parser for F02DC frames."""

    name = "F02DC"

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.collecting = False

    def reset(self) -> None:
        self.buffer.clear()
        self.collecting = False

    @staticmethod
    def _parse_frame(frame: bytes) -> str:
        """Validate an F02DC frame and return the U-Prox card number."""
        if len(frame) < 6:
            raise ValueError("Received incomplete F02DC frame, ignoring packet.")

        if frame[0] != PACKET_START or frame[-1] != PACKET_END:
            raise ValueError("Malformed F02DC frame boundaries, ignoring packet.")

        declared_length = frame[1]
        if declared_length != len(frame):
            raise ValueError(
                f"F02DC length mismatch: declared {declared_length}, actual {len(frame)}. Ignoring packet."
            )

        card_type = frame[2]
        payload = frame[3:-2]
        bcc = frame[-2]

        computed_bcc = 0
        for byte in frame[1:-2]:
            computed_bcc ^= byte

        if computed_bcc != bcc:
            raise ValueError(
                f"F02DC checksum mismatch: expected {bcc:02X}, computed {computed_bcc:02X}. Ignoring packet."
            )

        if not payload:
            raise ValueError("Empty F02DC payload, ignoring packet.")


        if card_type == 0x01:  # Mifare classic 1k
            payload = F02DCPacketReader._pad_payload(payload)
            return payload.hex().upper()

        if card_type == 0x11:  # 13.56 MHz NFC
            if len(payload) < 5:
                raise ValueError("F02DC NFC payload too short (expected at least 5 bytes).")
            order = (3, 4, 0, 1, 2)
            reordered = bytes(payload[i] for i in order if i < len(payload))
            return reordered.hex().upper()

        # Fallback: return raw payload for other card types
        return payload.hex().upper()

    @staticmethod
    def _pad_payload(payload: bytes) -> bytes:
        """Ensure payload is at least 5 bytes by left-padding with zeros."""
        if len(payload) >= 5:
            return payload
        return b"\x00" * (5 - len(payload)) + payload

    def feed(self, byte_value: int) -> ParsedResult | None:
        if not self.collecting:
            if byte_value == PACKET_START:
                self.reset()
                self.buffer.append(byte_value)
                self.collecting = True
            return None

        # Already collecting a frame; PACKET_START inside a frame is treated as data.
        self.buffer.append(byte_value)

        if byte_value == PACKET_END:
            try:
                frame_bytes = bytes(self.buffer)
                card_hex = self._parse_frame(frame_bytes)
            finally:
                self.reset()
            return card_hex, frame_bytes

        return None


async def serial_reader(
    queue: "asyncio.Queue[str]",
    serial_port: str,
    reader_factory: Callable[[], object],
) -> None:
    """Read RFID identifiers asynchronously using the selected protocol parser."""
    print(f"Connecting to reader at {serial_port}")

    while True:
        serial_conn = None
        reader = reader_factory()
        try:
            try:
                serial_conn = aioserial.AioSerial(port=serial_port, baudrate=BAUD_RATE, timeout=None)
                LOGGER.info("%s reader connected. Ready to accept cards.", getattr(reader, "name", "RFID"))
            except SERIAL_EXCEPTIONS as exc:
                print("Waiting for reader...", end="\r")
                LOGGER.debug(
                    "Unable to open %s (%s). Connect the reader and retrying in 1 second.",
                    serial_port,
                    exc,
                )
                await asyncio.sleep(0.5)
                continue

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
                try:
                    parsed = reader.feed(value)
                except ValueError as exc:
                    LOGGER.warning("%s", exc)
                    reader.reset()
                    continue

                if parsed is not None:
                    identifier, frame_bytes = parsed
                    LOGGER.debug(
                        "%s reader: Raw frame %s",
                        getattr(reader, "name", "RFID"),
                        " ".join(f"{b:02X}" for b in frame_bytes),
                    )
                    await queue.put(identifier)
                    LOGGER.debug("%s reader: Read identifier: %s", getattr(reader, "name", "RFID"), identifier)
        finally:
            if serial_conn is not None:
                serial_conn.close()
                LOGGER.debug("Serial port closed.")

        await asyncio.sleep(1)


async def identifier_dispatcher(queue: "asyncio.Queue[str]", ws_server: UProxWebSocketServer) -> None:
    """Wait for identifiers from the queue and broadcast them."""
    loop = asyncio.get_running_loop()
    last_seen: dict[str, float] = {}

    while True:
        identifier = await queue.get()
        now = loop.time()
        last_time = last_seen.get(identifier)
        if last_time is not None and now - last_time < IDENTIFIER_COOLDOWN_SECONDS:
            LOGGER.debug("Ignoring duplicate identifier %s read within %.2f seconds.", identifier, IDENTIFIER_COOLDOWN_SECONDS)
            continue
        last_seen[identifier] = now

        LOGGER.info("Card number: %s", identifier)
        await ws_server.broadcast(identifier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-D",
        "--device",
        default=DEFAULT_SERIAL_PORT,
        help=f"Serial device for the reader (default: {DEFAULT_SERIAL_PORT})",
    )
    parser.add_argument(
        "-R",
        "--reader",
        default=DEFAULT_READER,
        choices=["F02DC", "RDM6300"],
        help=f"Reader protocol to use (default: {DEFAULT_READER}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


async def main(serial_port: str, reader_choice: str) -> None:
    identifier_queue: "asyncio.Queue[str]" = asyncio.Queue()
    ws_server = UProxWebSocketServer(HOST, PORT)
    reader_factory: Callable[[], object]
    if reader_choice == "RDM6300":
        reader_factory = RDM6300PacketReader
    else:
        reader_factory = F02DCPacketReader

    try:
        async with serve(ws_server.handler, host=HOST, port=PORT):
            LOGGER.debug("WebSocket server running on ws://%s:%s", HOST, PORT)

            dispatcher_task = asyncio.create_task(identifier_dispatcher(identifier_queue, ws_server))
            serial_task = asyncio.create_task(serial_reader(identifier_queue, serial_port, reader_factory))
            await asyncio.gather(serial_task, dispatcher_task)
    except OSError as exc:
        LOGGER.error("Unable to open WebSocket server on ws://%s:%s: %s.\nCheck if you have another reader running", HOST, PORT, exc)
        await asyncio.sleep(5)


if __name__ == "__main__":
    cli_args = parse_args()
    configure_logging(getattr(logging, cli_args.log_level))
    try:
        asyncio.run(main(cli_args.device, cli_args.reader))
    except KeyboardInterrupt:
        LOGGER.info("Shutting down server.")
