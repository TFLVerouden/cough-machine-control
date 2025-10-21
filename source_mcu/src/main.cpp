/*
 * Cough Machine Control System
 *
 * Controls a solenoid valve for droplet experiments with precise timing.
 * Monitors pressure and environmental conditions.
 */

#include "MIKROE_4_20mA_RT_Click.h"
#include <Adafruit_DotStar.h>
#include <Adafruit_SHT4x.h>
#include <Arduino.h>

// ============================================================================
// DEBUG CONFIGURATION
// ============================================================================
// Set to 1 to enable debug messages, 0 to disable for maximum speed
#define DEBUG 1

#if DEBUG
  #define DEBUG_PRINT(x)    Serial.print(x)
  #define DEBUG_PRINTLN(x)  Serial.println(x)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
#endif

// ============================================================================
// PIN DEFINITIONS
// ============================================================================
const int PIN_VALVE = 7;     // MOSFET gate pin for solenoid valve control
const int PIN_CS_RCLICK = 2; // Chip select for R-Click pressure sensor (SPI)
const int PIN_TRIG = 9; // Trigger output for external device synchronization
// Note: PIN_DOTSTAR_DATA and PIN_DOTSTAR_CLK are already defined in variant.h

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
// LED CONFIGURATION
// ============================================================================
// DotStar RGB LED (using board's built-in DotStar on pins 8 and 6)
Adafruit_DotStar led(1, PIN_DOTSTAR_DATA, PIN_DOTSTAR_CLK, DOTSTAR_BGR);

// LED color definitions (avoiding pure red for laser goggle compatibility)
// Colors use BGR format: Blue, Green, Red
const uint32_t COLOR_IDLE = 0x001000;       // Dim green - system ready
const uint32_t COLOR_VALVE_OPEN = 0x00FF00; // Bright green - valve active
const uint32_t COLOR_ERROR = 0xFF6000;      // Orange - error state
const uint32_t COLOR_READING = 0xFF0040;    // Cyan - taking measurement
const uint32_t COLOR_OFF = 0x000000;        // Off

// ============================================================================
// LED HELPER FUNCTION
// ============================================================================

void setLedColor(uint32_t color) {
  // Set the DotStar LED to a specific color
  led.setPixelColor(0, color);
  led.show();
}

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

  // Initialize DotStar LED
  led.begin();
  led.setBrightness(255); // Set brightness to full
  led.show();             // Initialize all pixels to 'off'

  // Initialize serial communication at 115200 baud
  Serial.begin(115200);

  // Initialize pressure sensor
  R_click.begin();

  // Initialize the SHT4x temperature & humidity sensor
  if (!sht4.begin()) {
    Serial.println("Failed to find SHT4x sensor!");
    // Blink orange for fatal error
    while (1) {
      setLedColor(COLOR_ERROR);
      delay(200);
      setLedColor(COLOR_OFF);
      delay(200);
    }
  }

  // Configure SHT4x for high precision, no heater
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);

  // Show idle color to indicate system is ready
  setLedColor(COLOR_IDLE);
}

// ============================================================================
// VALVE AND SENSOR FUNCTIONS
// ============================================================================

void closeValve() {
  // Close valve using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_VALVE, LOW);
  PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_VALVE].ulPin);

  setLedColor(COLOR_IDLE);  // Return to idle color
  DEBUG_PRINTLN("!");       // Valve closed confirmation (debug only for speed)
}

void printError(const char *message) {
  // Print error message to serial for debugging
  Serial.print("ERROR: ");
  Serial.println(message);

  // Flash orange briefly to indicate error
  setLedColor(COLOR_ERROR);
  delay(300);
  setLedColor(COLOR_IDLE);
}

void readPressure(bool valveOpen) {
  // Read current pressure from R-Click sensor
  // Conversion formula: Pressure = 0.6249 * I[mA] - 2.4882
  // where I is the 4-20mA current output
  setLedColor(COLOR_READING); // Show color during reading
  Serial.print("P");
  Serial.print(0.6249 * R_click.get_EMA_mA() - 2.4882);
  Serial.println();
  // Restore LED color based on valve state
  setLedColor(valveOpen ? COLOR_VALVE_OPEN : COLOR_IDLE);
}

void readTemperature(bool valveOpen) {
  // Read temperature and relative humidity from SHT4x sensor
  setLedColor(COLOR_READING); // Show color during reading
  sensors_event_t humidity, temp;
  sht4.getEvent(&humidity, &temp);

  // Send temperature reading
  Serial.print("T");
  Serial.println(temp.temperature);

  // Send humidity reading
  Serial.print("RH");
  Serial.println(humidity.relative_humidity);

  // Restore LED color based on valve state
  setLedColor(valveOpen ? COLOR_VALVE_OPEN : COLOR_IDLE);
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
  // Close valve after duration (duration=0 means stay open)
  if (valveOpen && duration > 0 && (micros() - tick >= duration)) {
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
    command.trim(); // Remove any whitespace
    
    DEBUG_PRINT("CMD: ");
    DEBUG_PRINTLN(command);

    if (command == "O" || command.startsWith("O ")) {
      // Command: O or O <duration_ms>
      // O = open indefinitely, O <ms> = open for specified time

      if (command == "O") {
        duration = 0; // 0 means stay open
        DEBUG_PRINTLN("Opening valve indefinitely");
      } else {
      duration = 1000 * command.substring(2).toInt();
      DEBUG_PRINT("Opening valve for ");
      DEBUG_PRINT(duration);
      DEBUG_PRINTLN(" µs");
      }

        // Open valve and trigger
        PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTSET.reg =
            ((1 << g_APinDescription[PIN_VALVE].ulPin) |
             (1 << g_APinDescription[PIN_TRIG].ulPin));

        setLedColor(COLOR_VALVE_OPEN);
        valveOpen = true;
        performingTrigger = true;
      tick = micros();

    } else if (command == "C") {
      // Command: C
      // Manually close valve (override)
      closeValve();
      valveOpen = false;

    } else if (command == "P?") {
      // Command: P?
      // Read and return current pressure
      readPressure(valveOpen);

    } else if (command == "T?") {
      // Command: T?
      // Read and return temperature & humidity
      readTemperature(valveOpen);

    } else if (command == "?") {
      // Command: ?
      // Print help menu
      DEBUG_PRINTLN("\n=== Available Commands ===");
      DEBUG_PRINTLN("O      - Open valve indefinitely");
      DEBUG_PRINTLN("O <ms> - Open valve for <ms> milliseconds (e.g., O 100)");
      DEBUG_PRINTLN("C      - Close valve immediately");
      DEBUG_PRINTLN("P?     - Read pressure");
      DEBUG_PRINTLN("T?     - Read temperature & humidity");
      DEBUG_PRINTLN("S?     - System status");
      DEBUG_PRINTLN("?      - Show this help");

    } else if (command == "S?") {
      // Command: S?
      // Print system status (debug only)
      DEBUG_PRINTLN("\n=== System Status ===");
      DEBUG_PRINT("Valve: ");
      DEBUG_PRINTLN(valveOpen ? "OPEN" : "CLOSED");
      if (valveOpen) {
        DEBUG_PRINT("Time remaining: ");
        uint32_t elapsed = micros() - tick;
        if (elapsed < duration) {
          DEBUG_PRINT((duration - elapsed) / 1000);
          DEBUG_PRINTLN(" ms");
        }
      }
      DEBUG_PRINT("Trigger: ");
      DEBUG_PRINTLN(performingTrigger ? "ACTIVE" : "IDLE");
      DEBUG_PRINT("Pressure (raw): ");
      DEBUG_PRINT(R_click.get_EMA_mA());
      DEBUG_PRINTLN(" mA");
      DEBUG_PRINT("Uptime: ");
      DEBUG_PRINT(millis() / 1000);
      DEBUG_PRINTLN(" s");

    } else {
      // Unknown command
      printError("Unknown command");
    }
  }
}
