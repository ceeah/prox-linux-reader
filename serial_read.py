#!/usr/bin/env python3
# This shows hex values as read from serial port. Useful to debug new card reader packet formats
import serial
import time
import binascii
import sys

# Configure the serial port settings
# SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600

try:
    # Open the serial port
    # timeout=None makes the read() call a blocking call, waiting indefinitely for a byte
    # This is suitable for reading every single byte as it arrives.
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=None, 
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud rate.")
    ser.flushInput() # Clear the input buffer

    while True:
        # Read a single byte
        byte_data = ser.read(1)
        
        # Convert the byte to a hexadecimal string and print it
        # byte_data is a bytes object, .hex() method works directly on it
        hex_string = byte_data.hex() 
        
        # Print the hex value without a newline, followed by a space for readability
        print(f"{hex_string} ", end='', flush=True) 

except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nExiting program.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
finally:
    if 'ser' in locals() and ser.isOpen():
        ser.close()
        print("Serial port closed.")
