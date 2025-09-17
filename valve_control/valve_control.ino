#include <Arduino.h>
#include <Adafruit_SHT4x.h>
#include "MIKROE_4_20mA_RT_Click.h"

// Connections
const int ledPin = LED_BUILTIN;  // Use the built-in LED
const int mosfetPin = 7;         // Pin to control the MOSFET for valve control
const int csRClick = 2;          // Cable select pin for RClick pressure reading
const int trigPin = 10;      // Placeholder pin for pressure reading

// Valve opening logic parameters
int duration = 0;                // Variable to store the duration
bool valveOpen = false;          // Flag to check if the valve is open
unsigned long tickValve = 0;     // Variable to store the start time

// Exponential moving average (EMA) parameters & calibration of the R Click readings
const uint32_t EMA_INTERVAL = 500; // Desired oversampling interval [µs]
const float EMA_LP_FREQ = 200.;      // Low-pass filter cut-off frequency [Hz]
R_Click R_click(pressurePin, RT_Click_Calibration{3.99, 9.75, 795, 1943});

// Humidity+temperature sensor
Adafruit_SHT4x sht4;  // Declare the T+RH sensor object

enum State {
  IDLE,
  OPENING,
  CLOSING,
  ERROR
};

State currentState = IDLE;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(mosfetPin, OUTPUT);
  pinMode(csRClick, INPUT);
  digitalWrite(ledPin, LOW);
  digitalWrite(mosfetPin, LOW);
  digitalWrite(trigPin, LOW);
  Serial.begin(115200);
  R_click.begin();

  // Initialize the SHT4x sensor
  if (!sht4.begin()) {
    Serial.println("Failed to find SHT4x sensor!");
    while (1) delay(10);  // Halt execution if sensor is not found
  }

  // Set sensor precision and heater mode
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);
} 

void loop() {
  // Poll R-click board (pressure sensor)
  R_click.poll_EMA();

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    handleCommand(command);
  }

  switch (currentState) {
    case IDLE:
      // Do nothing
      break;

    case OPENING:
      if (millis() - startTime >= duration) {
        closeValve();
        currentState = IDLE;
      }
      break;

    case CLOSING:
      closeValve();
      currentState = IDLE;
      break;

    case ERROR:
      blinkError();
      currentState = IDLE;
      break;
  }
}

void handleCommand(String command) {
  if (command.startsWith("O")) {
    duration = command.substring(2).toInt();
    if (duration > 0) {
      openValve();
      currentState = OPENING;
    } else {
      currentState = ERROR;
    }

  } else if (command == "C") {
    if (valveOpen) {
      currentState = CLOSING;
    }

  } else if (command == "P?") {
    readPressure();

  } else if (command == "T?") {
    readTemperature();
  } else {
    currentState = ERROR;
  }
}

void openValve() {
  digitalWrite(ledPin, HIGH);
  digitalWrite(mosfetPin, HIGH);
  valveOpen = true;
  tickValve = millis();
}

void closeValve() {
  digitalWrite(ledPin, LOW);
  digitalWrite(mosfetPin, LOW);
  valveOpen = false;
  Serial.println("!");
}

void blinkError() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(ledPin, HIGH);
    delay(100);
    digitalWrite(ledPin, LOW);
    delay(100);
  }
}

void readPressure() {
  // Reads out R-click board and converts to pressure
  Serial.print("P");
  Serial.print(0.6249*R_click.get_EMA_mA() - 2.4882);
  Serial.println();
}

void readTemperature() {
  // Reads Temperature and RH on command;
  sensors_event_t humidity, temp;
  
  // Read temperature and humidity
  sht4.getEvent(&humidity, &temp);
  
  Serial.print("T");
  Serial.println(temp.temperature);
  
  Serial.print("RH");
  Serial.println(humidity.relative_humidity);
}
