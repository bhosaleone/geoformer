/*
 * GeoFormer Nano: Quantized Thermodynamic Attractor Convergence (Q-TAC) Engine
 * Designed for 8-bit Microcontrollers (ATmega328P / Arduino Uno)
 * 
 * Target Specs: 16 MHz AVR, 2 KB SRAM, 32 KB Flash
 * Author: Shrikant Bhosale (Atmabhan Pandit), GMC Sambhajinagar
 * Organization: SetMD / TWIST POOL Labs / NanoCERN Unit
 * License: GNU Affero General Public License v3.0 (AGPLv3)
 */

#define F_CPU 16000000UL
#define BAUD 115200

#include <avr/io.h>
#include <avr/pgmspace.h>
#include <util/delay.h>

#define NUM_NODES 16

// Conductance lattice in SRAM: 16x16 = 256 bytes
static uint8_t W[NUM_NODES][NUM_NODES];

// Directed transition probabilities: 16x16 = 256 bytes (0..255 where 255 = 1.0)
static uint8_t P[NUM_NODES][NUM_NODES];

// Node activations (16 + 16 = 32 bytes)
static uint8_t h[NUM_NODES];
static uint8_t h_next[NUM_NODES];

#define LED_PIN (1 << PB5)

/* ==================== UART DRIVER WITH PROGMEM ==================== */

static void uart_init(void) {
    UCSR0A = (1 << U2X0);
    uint16_t ubrr_val = (F_CPU / (8UL * BAUD)) - 1; // 16 for 115200 baud
    UBRR0H = (uint8_t)(ubrr_val >> 8);
    UBRR0L = (uint8_t)ubrr_val;
    UCSR0B = (1 << RXEN0) | (1 << TXEN0);
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

static void uart_putchar(char c) {
    while (!(UCSR0A & (1 << UDRE0)));
    UDR0 = c;
}

static void uart_puts_P(const char *progmem_s) {
    char c;
    while ((c = pgm_read_byte(progmem_s++))) {
        if (c == '\n') uart_putchar('\r');
        uart_putchar(c);
    }
}

static void uart_putnum(uint32_t val) {
    char buf[12];
    int i = 0;
    if (val == 0) {
        uart_putchar('0');
        return;
    }
    while (val > 0) {
        buf[i++] = (val % 10) + '0';
        val /= 10;
    }
    while (i > 0) {
        uart_putchar(buf[--i]);
    }
}

static char uart_getchar(void) {
    while (!(UCSR0A & (1 << RXC0)));
    return UDR0;
}

static uint8_t uart_available(void) {
    return (UCSR0A & (1 << RXC0)) != 0;
}

/* ==================== TIMER1 FOR LATENCY BENCHMARKING ==================== */

static void timer_init(void) {
    TCCR1A = 0;
    TCCR1B = 0;
}

static void timer_start(void) {
    TCNT1 = 0;
    TCCR1B = (1 << CS11) | (1 << CS10); // prescaler 64: 4 us per tick
}

static uint16_t timer_stop_us(void) {
    TCCR1B = 0;
    uint16_t ticks = TCNT1;
    return ticks * 4;
}

/* ==================== GEOFORMER QUANTIZED LATTICE & Q-TAC ==================== */

static void carve_geodesic(uint8_t src, uint8_t dst, uint8_t weight) {
    if (src >= NUM_NODES || dst >= NUM_NODES) return;
    uint16_t new_w = (uint16_t)W[src][dst] + weight;
    if (new_w > 15) new_w = 15;
    W[src][dst] = (uint8_t)new_w;
}

static void update_transitions(void) {
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        uint16_t row_sum = 0;
        for (uint8_t j = 0; j < NUM_NODES; j++) {
            row_sum += W[i][j];
        }
        for (uint8_t j = 0; j < NUM_NODES; j++) {
            if (row_sum > 0) {
                P[i][j] = (uint8_t)(((uint32_t)W[i][j] * 255) / row_sum);
            } else {
                P[i][j] = 0;
            }
        }
    }
}

static void qtac_step(void) {
    for (uint8_t j = 0; j < NUM_NODES; j++) {
        uint32_t acc = 0;
        for (uint8_t i = 0; i < NUM_NODES; i++) {
            if (h[i] > 0 && P[i][j] > 0) {
                acc += (uint32_t)h[i] * P[i][j];
            }
        }
        h_next[j] = (uint8_t)(acc / 255);
    }
    for (uint8_t j = 0; j < NUM_NODES; j++) {
        h[j] = h_next[j];
    }
}

static void inject_pulse(uint8_t node) {
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        h[i] = 0;
    }
    if (node < NUM_NODES) {
        h[node] = 255;
    }
}

static uint8_t get_argmax(void) {
    uint8_t best_node = 0;
    uint8_t max_val = 0;
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        if (h[i] > max_val) {
            max_val = h[i];
            best_node = i;
        }
    }
    return best_node;
}

static void print_activations(uint8_t step) {
    uart_puts_P(PSTR("  tau="));
    uart_putnum(step);
    uart_puts_P(PSTR(": [ "));
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        uint16_t pct = ((uint32_t)h[i] * 100) / 255;
        if (pct > 0) {
            uart_puts_P(PSTR("N"));
            uart_putnum(i);
            uart_puts_P(PSTR("="));
            uart_putnum(pct);
            uart_puts_P(PSTR("% "));
        }
    }
    uart_puts_P(PSTR("]\n"));
}

/* ==================== DEMONSTRATION & BENCHMARKS ==================== */

static void init_default_graph(void) {
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        for (uint8_t j = 0; j < NUM_NODES; j++) {
            W[i][j] = 0;
        }
    }
    // 3-hop chain: 0 -> 1 -> 2 -> 3
    carve_geodesic(0, 1, 8);
    carve_geodesic(1, 2, 8);
    carve_geodesic(2, 3, 8);

    // Branching crossroad: 1 -> 4 -> 5
    carve_geodesic(1, 4, 4);
    carve_geodesic(4, 5, 8);

    // Clinical emergency triage chain:
    // Node 10 (Shock) -> Node 11 (Hypotension) -> Node 12 (Emergency Vasopressors)
    carve_geodesic(10, 11, 10);
    carve_geodesic(11, 12, 10);

    update_transitions();
}

static void run_transitive_query(uint8_t src_node, uint8_t max_tau) {
    uart_puts_P(PSTR("\n--------------------------------------------------\n"));
    uart_puts_P(PSTR(">>> Injected Source Pulse on Node "));
    uart_putnum(src_node);
    uart_puts_P(PSTR("\n--------------------------------------------------\n"));
    
    PORTB |= LED_PIN;
    timer_start();
    
    inject_pulse(src_node);
    print_activations(0);

    for (uint8_t tau = 1; tau <= max_tau; tau++) {
        qtac_step();
        print_activations(tau);
    }

    uint16_t elapsed_us = timer_stop_us();
    PORTB &= ~LED_PIN;

    uint8_t attractor = get_argmax();
    uint16_t confidence = ((uint32_t)h[attractor] * 100) / 255;

    uart_puts_P(PSTR(">>> Attractor Settled on Node "));
    uart_putnum(attractor);
    uart_puts_P(PSTR(" (Confidence: "));
    uart_putnum(confidence);
    uart_puts_P(PSTR("%)\n"));
    uart_puts_P(PSTR(">>> Inference Latency: "));
    uart_putnum(elapsed_us);
    uart_puts_P(PSTR(" us\n"));
}

static void print_conductance_matrix(void) {
    uart_puts_P(PSTR("\n=== 4-BIT METRIC CONDUCTANCE LATTICE W (16x16) ===\n     "));
    for (uint8_t j = 0; j < NUM_NODES; j++) {
        if (j < 10) uart_putchar(' ');
        uart_putnum(j);
        uart_putchar(' ');
    }
    uart_puts_P(PSTR("\n"));
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        if (i < 10) uart_putchar(' ');
        uart_putnum(i);
        uart_puts_P(PSTR(": "));
        for (uint8_t j = 0; j < NUM_NODES; j++) {
            if (W[i][j] == 0) {
                uart_puts_P(PSTR(" . "));
            } else {
                if (W[i][j] < 10) uart_putchar(' ');
                uart_putnum(W[i][j]);
                uart_putchar(' ');
            }
        }
        uart_puts_P(PSTR("\n"));
    }
    uart_puts_P(PSTR("===================================================\n"));
}

static void run_1000_cycle_benchmark(void) {
    uart_puts_P(PSTR("\n[BENCHMARK] Executing 1,000 Q-TAC Heat Relaxation Steps...\n"));
    inject_pulse(0);
    
    PORTB |= LED_PIN;
    timer_start();
    for (uint16_t c = 0; c < 1000; c++) {
        qtac_step();
        if ((c & 0x07) == 0) h[0] += 16;
    }
    uint16_t total_us = timer_stop_us();
    PORTB &= ~LED_PIN;

    uart_puts_P(PSTR(">>> 1,000 Steps Completed in: "));
    uart_putnum(total_us);
    uart_puts_P(PSTR(" us\n"));
    uart_puts_P(PSTR(">>> Average Latency per Step: "));
    uart_putnum(total_us / 1000);
    uart_puts_P(PSTR("."));
    uart_putnum((total_us % 1000) / 100);
    uart_puts_P(PSTR(" us\n"));
    if (total_us > 0) {
        uint32_t steps_per_sec = (1000UL * 1000000UL) / total_us;
        uart_puts_P(PSTR(">>> Throughput: "));
        uart_putnum(steps_per_sec);
        uart_puts_P(PSTR(" steps/sec on 16 MHz 8-bit AVR!\n"));
    }
}

/* ==================== MAIN ENTRY ==================== */

int main(void) {
    DDRB |= LED_PIN;
    PORTB &= ~LED_PIN;

    uart_init();
    timer_init();
    init_default_graph();

    _delay_ms(100);

    uart_puts_P(PSTR("\n\n"));
    uart_puts_P(PSTR("====================================================\n"));
    uart_puts_P(PSTR("    GEOFORMER NANO: QUANTIZED PROPAGATION GEOMETRY  \n"));
    uart_puts_P(PSTR("    Running on Atmel ATmega328P @ 16 MHz (Uno)      \n"));
    uart_puts_P(PSTR("    Author: Shrikant Bhosale (Atmabhan Pandit)      \n"));
    uart_puts_P(PSTR("    Zenodo DOI: 10.5281/zenodo.23730305             \n"));
    uart_puts_P(PSTR("====================================================\n"));
    uart_puts_P(PSTR("Commands:\n"));
    uart_puts_P(PSTR("  [1] Query 2-Hop Route: Node 0 -> Node 2 (tau=2)\n"));
    uart_puts_P(PSTR("  [2] Query 3-Hop Route: Node 0 -> Node 3 (tau=3)\n"));
    uart_puts_P(PSTR("  [3] Dynamic In-Context Plasticity: Carve 0->3 Highway\n"));
    uart_puts_P(PSTR("  [4] Inspect 16x16 Metric Conductance Lattice\n"));
    uart_puts_P(PSTR("  [5] Run 1,000-Step Hardware Benchmark & Latency Audit\n"));
    uart_puts_P(PSTR("  [6] Medical Triage: Node 10 (Shock) -> Node 12 (Rx)\n"));
    uart_puts_P(PSTR("====================================================\n\n"));

    run_transitive_query(0, 2);

    while (1) {
        if (uart_available()) {
            char cmd = uart_getchar();
            switch (cmd) {
                case '1':
                    run_transitive_query(0, 2);
                    break;
                case '2':
                    run_transitive_query(0, 3);
                    break;
                case '3':
                    uart_puts_P(PSTR("\n[PLASTICITY] Carving Geodesic Highway: 0 ==> 3 (+12 conductance)\n"));
                    carve_geodesic(0, 3, 12);
                    update_transitions();
                    uart_puts_P(PSTR("[PLASTICITY] Lattice rewired. Re-querying at tau=1:\n"));
                    run_transitive_query(0, 1);
                    break;
                case '4':
                    print_conductance_matrix();
                    break;
                case '5':
                    run_1000_cycle_benchmark();
                    break;
                case '6':
                    uart_puts_P(PSTR("\n[CLINICAL TRIAGE] Pulse on Node 10 (Shock)...\n"));
                    run_transitive_query(10, 2);
                    break;
                default:
                    if (cmd != '\r' && cmd != '\n') {
                        uart_puts_P(PSTR("\nType 1, 2, 3, 4, 5, or 6.\n"));
                    }
                    break;
            }
        }
        _delay_ms(10);
    }

    return 0;
}
