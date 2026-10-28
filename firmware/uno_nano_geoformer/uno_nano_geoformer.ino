/*
 * GeoFormer Nano: Quantized Thermodynamic Attractor Convergence (Q-TAC) Engine
 * Designed for Arduino Uno (ATmega328P @ 16 MHz)
 * 
 * Target: Arduino Uno / Nano (2 KB SRAM, 32 KB Flash)
 * Author: Shrikant Bhosale (Atmabhan Pandit), GMC Sambhajinagar
 * Organization: SetMD / TWIST POOL Labs / NanoCERN Unit
 * License: GNU Affero General Public License v3.0 (AGPLv3)
 * Zenodo DOI: 10.5281/zenodo.23730305
 */

#define NUM_NODES 16
#define FP_SHIFT 8
#define FP_ONE (1 << FP_SHIFT)  // 256 = 1.0 in fixed-point

// 4-bit Metric Conductance Lattice (16x16 = 256 bytes SRAM)
uint8_t W[NUM_NODES][NUM_NODES];

// Directed Row-Normalized Transition Matrix (16x16*2 = 512 bytes SRAM)
uint16_t P[NUM_NODES][NUM_NODES];

// Attractor Activation Vectors (16*2 = 32 bytes)
uint16_t h[NUM_NODES];
uint16_t h_next[NUM_NODES];

const int LED_PIN = 13;

void carve_geodesic(uint8_t src, uint8_t dst, uint8_t weight) {
    if (src >= NUM_NODES || dst >= NUM_NODES) return;
    uint16_t new_w = (uint16_t)W[src][dst] + weight;
    if (new_w > 15) new_w = 15; // 4-bit saturation
    W[src][dst] = (uint8_t)new_w;
}

void update_transitions() {
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        uint16_t row_sum = 0;
        for (uint8_t j = 0; j < NUM_NODES; j++) row_sum += W[i][j];
        for (uint8_t j = 0; j < NUM_NODES; j++) {
            if (row_sum > 0) {
                P[i][j] = ((uint32_t)W[i][j] * FP_ONE) / row_sum;
            } else {
                P[i][j] = 0;
            }
        }
    }
}

void qtac_step() {
    for (uint8_t j = 0; j < NUM_NODES; j++) {
        uint32_t acc = 0;
        for (uint8_t i = 0; i < NUM_NODES; i++) {
            if (h[i] > 0 && P[i][j] > 0) {
                acc += (uint32_t)h[i] * P[i][j];
            }
        }
        h_next[j] = (uint16_t)(acc >> FP_SHIFT);
    }
    for (uint8_t j = 0; j < NUM_NODES; j++) h[j] = h_next[j];
}

void inject_pulse(uint8_t node) {
    for (uint8_t i = 0; i < NUM_NODES; i++) h[i] = 0;
    if (node < NUM_NODES) h[node] = FP_ONE;
}

uint8_t get_argmax() {
    uint8_t best = 0;
    uint16_t max_val = 0;
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        if (h[i] > max_val) {
            max_val = h[i];
            best = i;
        }
    }
    return best;
}

void print_activations(uint8_t step) {
    Serial.print(F("  Step tau="));
    Serial.print(step);
    Serial.print(F(": [ "));
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        uint16_t pct = ((uint32_t)h[i] * 100) >> FP_SHIFT;
        if (pct > 0) {
            Serial.print(F("Node"));
            Serial.print(i);
            Serial.print(F("="));
            Serial.print(pct);
            Serial.print(F("% "));
        }
    }
    Serial.println(F("]"));
}

void init_default_graph() {
    memset(W, 0, sizeof(W));
    
    // Multi-Hop Transitive Chains:
    // Chain 1: 0 -> 1 -> 2 -> 3 (3-hop composition)
    carve_geodesic(0, 1, 8);
    carve_geodesic(1, 2, 8);
    carve_geodesic(2, 3, 8);

    // Crossroad branching from node 1 -> 4 -> 5
    carve_geodesic(1, 4, 4);
    carve_geodesic(4, 5, 8);

    // Chain 2: 6 -> 7 -> 8 -> 9
    carve_geodesic(6, 7, 8);
    carve_geodesic(7, 8, 8);
    carve_geodesic(8, 9, 8);

    // Chain 3 (Clinical Casualty Triage):
    // Node 10 (Shock) -> Node 11 (Severe Hypotension) -> Node 12 (Emergency Vasopressors)
    carve_geodesic(10, 11, 10);
    carve_geodesic(11, 12, 10);

    update_transitions();
}

void run_transitive_query(uint8_t src_node, uint8_t max_tau) {
    Serial.println(F("\n--------------------------------------------------"));
    Serial.print(F(">>> Injected Source Pulse on Node "));
    Serial.println(src_node);
    Serial.println(F("--------------------------------------------------"));
    
    digitalWrite(LED_PIN, HIGH);
    unsigned long start_us = micros();
    
    inject_pulse(src_node);
    print_activations(0);

    for (uint8_t tau = 1; tau <= max_tau; tau++) {
        qtac_step();
        print_activations(tau);
    }

    unsigned long elapsed_us = micros() - start_us;
    digitalWrite(LED_PIN, LOW);

    uint8_t attractor = get_argmax();
    uint16_t confidence = ((uint32_t)h[attractor] * 100) >> FP_SHIFT;

    Serial.print(F(">>> RESULT: Attractor Settled on Node "));
    Serial.print(attractor);
    Serial.print(F(" (Confidence: "));
    Serial.print(confidence);
    Serial.println(F("%)"));
    Serial.print(F(">>> Inference Latency: "));
    Serial.print(elapsed_us);
    Serial.println(F(" microseconds (us)"));
}

void print_conductance_matrix() {
    Serial.println(F("\n=== 4-BIT METRIC CONDUCTANCE LATTICE W (16x16) ==="));
    Serial.print(F("     "));
    for (uint8_t j = 0; j < NUM_NODES; j++) {
        if (j < 10) Serial.print(F(" "));
        Serial.print(j);
        Serial.print(F(" "));
    }
    Serial.println();
    for (uint8_t i = 0; i < NUM_NODES; i++) {
        if (i < 10) Serial.print(F(" "));
        Serial.print(i);
        Serial.print(F(": "));
        for (uint8_t j = 0; j < NUM_NODES; j++) {
            if (W[i][j] == 0) {
                Serial.print(F(" . "));
            } else {
                if (W[i][j] < 10) Serial.print(F(" "));
                Serial.print(W[i][j]);
                Serial.print(F(" "));
            }
        }
        Serial.println();
    }
    Serial.println(F("==================================================="));
}

void run_1000_cycle_benchmark() {
    Serial.println(F("\n[BENCHMARK] Executing 1,000 Q-TAC Heat Relaxation Steps..."));
    inject_pulse(0);
    
    digitalWrite(LED_PIN, HIGH);
    unsigned long start_us = micros();
    for (uint16_t c = 0; c < 1000; c++) {
        qtac_step();
        if ((c & 0x07) == 0) h[0] += 16;
    }
    unsigned long total_us = micros() - start_us;
    digitalWrite(LED_PIN, LOW);

    Serial.print(F(">>> 1,000 Steps Completed in: "));
    Serial.print(total_us);
    Serial.println(F(" us"));
    Serial.print(F(">>> Average Latency per Q-TAC Step: "));
    Serial.print(total_us / 1000.0, 2);
    Serial.println(F(" us"));
    if (total_us > 0) {
        float steps_per_sec = (1000.0 * 1000000.0) / total_us;
        Serial.print(F(">>> Throughput: "));
        Serial.print(steps_per_sec, 0);
        Serial.println(F(" steps/second on 16 MHz 8-bit AVR!"));
    }
}

void print_menu() {
    Serial.println(F("\n===================================================="));
    Serial.println(F("    GEOFORMER NANO: QUANTIZED PROPAGATION GEOMETRY  "));
    Serial.println(F("    Running on Atmel ATmega328P @ 16 MHz (Uno)      "));
    Serial.println(F("    Author: Shrikant Bhosale (Atmabhan Pandit)      "));
    Serial.println(F("    Zenodo DOI: 10.5281/zenodo.23730305             "));
    Serial.println(F("===================================================="));
    Serial.println(F("Commands:"));
    Serial.println(F("  [1] Query 2-Hop Transitive Route: Node 0 -> Node 2 (tau=2)"));
    Serial.println(F("  [2] Query 3-Hop Deep Route: Node 0 -> Node 3 (tau=3)"));
    Serial.println(F("  [3] Dynamic In-Context Adaptation: Carve Geodesic Highway (0->3)"));
    Serial.println(F("  [4] Inspect 16x16 Metric Conductance Lattice"));
    Serial.println(F("  [5] Run 1,000-Step Hardware Benchmark & Latency Audit"));
    Serial.println(F("  [6] Medical Triage Deduction: Node 10 (Shock) -> Node 12 (Rx)"));
    Serial.println(F("====================================================\n"));
}

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.begin(115200);
    while (!Serial && millis() < 3000); // Wait for serial monitor

    init_default_graph();
    print_menu();

    // Initial 2-hop run on boot
    run_transitive_query(0, 2);
}

void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();
        switch (cmd) {
            case '1':
                run_transitive_query(0, 2);
                break;
            case '2':
                run_transitive_query(0, 3);
                break;
            case '3':
                Serial.println(F("\n[ADAPTATION] Carving direct Geodesic Highway: Node 0 ==> Node 3 (+12 conductance)"));
                carve_geodesic(0, 3, 12);
                update_transitions();
                Serial.println(F("[ADAPTATION] Conductance updated in SRAM. Re-evaluating query:"));
                run_transitive_query(0, 1);
                break;
            case '4':
                print_conductance_matrix();
                break;
            case '5':
                run_1000_cycle_benchmark();
                break;
            case '6':
                Serial.println(F("\n[CLINICAL TRIAGE] Pulse injected at Node 10 (Emergency Shock)..."));
                run_transitive_query(10, 2);
                break;
            case '\r':
            case '\n':
                break;
            default:
                Serial.println(F("Unknown command. Type 1, 2, 3, 4, 5, or 6."));
                break;
        }
    }
}
