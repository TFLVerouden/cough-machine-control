#include <SPI.h>

#define CS_PIN 10  // Chip Select pin (adjust if needed)

void setup() {
  Serial.begin(115200);
  SPI.begin();                // Initialize SPI
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH); // Deselect the Click board
}

uint16_t readSensorData() {
  digitalWrite(CS_PIN, LOW); // Select Click board
  delayMicroseconds(10);     // Short delay for stability

  uint16_t data = SPI.transfer16(0x0000); // Read 16-bit data from Click board

  digitalWrite(CS_PIN, HIGH); // Deselect Click board
  return data;
}

void loop() {
  uint16_t sensorData = readSensorData();
  Serial.print("Raw Sensor Data: ");
  Serial.println(sensorData);

  // Convert raw data to pressure based on sensor scaling
  float pressure = (sensorData / 65535.0) * 10.0; // Example conversion
  Serial.print("Pressure: ");
  Serial.print(pressure);
  Serial.println(" bar");

  delay(500);
}
