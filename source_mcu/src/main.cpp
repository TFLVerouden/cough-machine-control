/*
 * Cough Machine Control System
 *
 * Controls a solenoid valve for droplet experiments with precise timing.
 * Monitors pressure and environmental conditions.
 */

#include "MIKROE_4_20mA_RT_Click.h"
#include <Adafruit_SHT4x.h>
#include <Arduino.h>

// ============================================================================
// PIN DEFINITIONS
// ============================================================================
const int PIN_VALVE = 7;     // MOSFET gate pin for solenoid valve control
const int PIN_CS_RCLICK = 2; // Chip select for R-Click pressure sensor (SPI)
const int PIN_TRIG = 9; // Trigger output for external device synchronization

// ============================================================================
// TIMING PARAMETERS
// ============================================================================
const uint32_t TRIGGER_WIDTH = 10000; // Trigger pulse width [µs] (10ms)
uint32_t tick = 0;                    // Timestamp for timing events [µs]

// ============================================================================
// SENSOR CONFIGURATION
// ============================================================================
// Pressure sensor (4-20mA R-Click) with exponential moving average filtering
const uint32_t EMA_INTERVAL = 500; // Sampling interval for EMA [µs]
const float EMA_LP_FREQ = 200.;    // Low-pass filter cutoff frequency [Hz]
// Initialize with calibration values: p1_mA, p2_mA, p1_bitval, p2_bitval
R_Click R_click(PIN_CS_RCLICK, RT_Click_Calibration{3.99, 9.75, 795, 1943});

// Temperature & humidity sensor (SHT4x I2C)
Adafruit_SHT4x sht4;

// ============================================================================
// INITIALIZATION
// ============================================================================

void setup() {
  // Configure pin modes
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_VALVE, OUTPUT);
  pinMode(PIN_CS_RCLICK, INPUT);
  pinMode(PIN_TRIG, OUTPUT);

  // Set all outputs to safe initial state (off)
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_VALVE, LOW);
  digitalWrite(PIN_TRIG, LOW);

  // Initialize serial communication at 115200 baud
  Serial.begin(115200);

  // Initialize pressure sensor
  R_click.begin();

  // Initialize the SHT4x temperature & humidity sensor
  if (!sht4.begin()) {
    Serial.println("Failed to find SHT4x sensor!");
    while (1)
      delay(10); // Halt execution if sensor is not found
  }

  // Configure SHT4x for high precision, no heater
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void closeValve() {
  // Close valve using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_VALVE, LOW);
  PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_VALVE].ulPin);

  digitalWrite(PIN_LED, LOW); // Turn off LED indicator
  Serial.println("!");        // Send valve closed confirmation
}

void printError(const char *message) {
  // Print error message to serial for debugging
  Serial.print("ERROR: ");
  Serial.println(message);
}

void readPressure() {
  // Read current pressure from R-Click sensor
  // Conversion formula: Pressure = 0.6249 * I[mA] - 2.4882
  // where I is the 4-20mA current output
  Serial.print("P");
  Serial.print(0.6249 * R_click.get_EMA_mA() - 2.4882);
  Serial.println();
}

void readTemperature() {
  // Read temperature and relative humidity from SHT4x sensor
  sensors_event_t humidity, temp;
  sht4.getEvent(&humidity, &temp);

  // Send temperature reading
  Serial.print("T");
  Serial.println(temp.temperature);

  // Send humidity reading
  Serial.print("RH");
  Serial.println(humidity.relative_humidity);
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Static variables persist across loop iterations (like globals, but scoped)
  static uint32_t duration = 0;          // How long valve should stay open [µs]
  static bool valveOpen = false;         // Tracks if valve is currently open
  static bool performingTrigger = false; // Tracks if trigger pulse is active

  // -------------------------------------------------------------------------
  // Handle trigger pulse timing
  // -------------------------------------------------------------------------
  // The trigger pulse is a short signal sent to external equipment when the
  // valve opens. It turns off after TRIGGER_WIDTH microseconds.
  if (performingTrigger && (micros() - tick >= TRIGGER_WIDTH)) {
    // Turn off trigger using direct PORT register access
    // Equivalent to digitalWrite(PIN_TRIG, LOW);
    PORT->Group[g_APinDescription[PIN_TRIG].ulPort].OUTCLR.reg =
        (1 << g_APinDescription[PIN_TRIG].ulPin);
    performingTrigger = false;
  }

  // -------------------------------------------------------------------------
  // Handle valve timing
  // -------------------------------------------------------------------------
  // Close valve automatically after the specified duration has elapsed
  if (valveOpen && (micros() - tick >= duration)) {
    closeValve();
    valveOpen = false;
  }

  // -------------------------------------------------------------------------
  // Update pressure sensor
  // -------------------------------------------------------------------------
  // Must be called regularly to maintain the exponential moving average
  R_click.poll_EMA();

  // -------------------------------------------------------------------------
  // Process serial commands
  // -------------------------------------------------------------------------
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');

    if (command.startsWith("O")) {
      // Command: O<duration_ms>
      // Opens valve for specified duration in milliseconds
      // Example: "O100" opens valve for 100ms

      // Extract duration and convert ms to µs
      duration = 1000 * command.substring(2).toInt();

      if (duration > 0) {
        // Open both valve and trigger simultaneously using PORT register
        // Eq. to: digitalWrite(PIN_VALVE, HIGH); digitalWrite(PIN_TRIG, HIGH);
        PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTSET.reg =
            ((1 << g_APinDescription[PIN_VALVE].ulPin) |
             (1 << g_APinDescription[PIN_TRIG].ulPin));

        digitalWrite(PIN_LED, HIGH); // Turn on LED indicator
        valveOpen = true;
        performingTrigger = true;
        tick = micros(); // Record start time
      } else {
        printError("Invalid duration");
      }

    } else if (command == "C") {
      // Command: C
      // Manually close valve (override)
      closeValve();
      valveOpen = false;

    } else if (command == "P?") {
      // Command: P?
      // Read and return current pressure
      readPressure();

    } else if (command == "T?") {
      // Command: T?
      // Read and return temperature & humidity
      readTemperature();

    } else {
      // Unknown command
      printError("Unknown command");
    }
  }
}
