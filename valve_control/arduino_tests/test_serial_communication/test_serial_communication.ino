const int ledPin = LED_BUILTIN;  // Use the built-in LED

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
    if (command == "ON") {
      digitalWrite(ledPin, HIGH);  // Turn the LED on
      Serial.println("LED is ON");
    } else if (command == "OFF") {
      digitalWrite(ledPin, LOW);  // Turn the LED off
      Serial.println("LED is OFF");
    } else {
      Serial.println("Error: Unknown command");
    }
  }
}