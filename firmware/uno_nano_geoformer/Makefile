MCU = atmega328p
F_CPU = 16000000UL
PORT = /dev/ttyACM0
BAUD = 115200
PROGRAMMER = arduino

CC = avr-gcc
OBJCOPY = avr-objcopy
AVRDUDE = avrdude

CFLAGS = -Wall -Os -mmcu=$(MCU) -DF_CPU=$(F_CPU) -std=gnu99 -funsigned-char -funsigned-bitfields -fpack-struct -fshort-enums

all: main.hex

main.elf: main.c
	$(CC) $(CFLAGS) -o $@ $<

main.hex: main.elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@
	@echo "=== Firmware Size ==="
	@avr-size --mcu=$(MCU) --format=avr main.elf

flash: main.hex
	$(AVRDUDE) -v -p $(MCU) -c $(PROGRAMMER) -P $(PORT) -b $(BAUD) -D -U flash:w:$<:i

clean:
	rm -f main.elf main.hex

.PHONY: all flash clean
