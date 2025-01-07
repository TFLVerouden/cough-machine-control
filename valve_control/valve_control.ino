const int ledPin = LED_BUILTIN;  // Use the built-in LED
int duration = 0;  // Variable to store the duration
bool timeSet = false;  // Flag to check if time has been set

// TODO: Precede each print with "ARDUINO SAYS:"

void setup() {
  pinMode(ledPin, OUTPUT);  // Set the LED pin as an output
  digitalWrite(ledPin, LOW);  // Ensure the LED is off initially
  Serial.begin(9600);  // Start the serial communication
  Serial.println("Arduino is ready");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    Serial.print("Received command: ");
    Serial.println(command);
    if (command.startsWith("TIME")) {
      duration = command.substring(5).toInt();
      Serial.print("Setting time to: ");
      Serial.print(duration);
      Serial.println(" ms");
      timeSet = true;
    } else if (command == "OPEN") {
      if (timeSet) {
        Serial.println("Valve is OPEN");
        digitalWrite(ledPin, HIGH);  // Turn the LED on
        delay(duration);  // Keep the LED on for the specified duration
        digitalWrite(ledPin, LOW);  // Turn the LED off
        Serial.println("Valve is CLOSED");
      } else {
        Serial.println("Error: TIME not set");
        // Blink the LED to indicate an error
        for (int i = 0; i < 5; i++) {
          digitalWrite(ledPin, HIGH);
          delay(100);
          digitalWrite(ledPin, LOW);
          delay(100);
        }
      }
    } else if (command == "RESET") {
      Serial.println("Resetting time");
      timeSet = false;
    } else {
      Serial.println("Error: Unknown command");
    }
  }
}