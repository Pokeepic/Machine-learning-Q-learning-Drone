// --- PINS ---
int motor1pin1 = 2;
int motor1pin2 = 3;
int motor2pin1 = 4;
int motor2pin2 = 5;

// --- TUNING ---
int DRIVE_TIME = 500;  // How long to move forward (ms) per grid cell
int TURN_TIME = 350;   // How long to spin (ms) for 90 degrees

void setup() {
  Serial.begin(9600);
  pinMode(motor1pin1, OUTPUT);
  pinMode(motor1pin2, OUTPUT);
  pinMode(motor2pin1, OUTPUT);
  pinMode(motor2pin2, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'F') {
      moveForward();
      delay(DRIVE_TIME);
    } 
    else if (cmd == 'L') {
      turnLeft();
      delay(TURN_TIME);
    } 
    else if (cmd == 'R') {
      turnRight();
      delay(TURN_TIME);
    }

    stopMotors();
    delay(200); // Stabilization pause
    
    // Handshake: Tell Python we are done
    Serial.print('K'); 
  }
}

// --- MOVEMENT FUNCTIONS ---

void moveForward() {
  // Left Motor Forward
  digitalWrite(motor1pin1, HIGH);
  digitalWrite(motor1pin2, LOW);
  // Right Motor Forward
  digitalWrite(motor2pin1, HIGH);
  digitalWrite(motor2pin2, LOW);
}

void turnLeft() {
  // Left Motor Back
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, HIGH);
  // Right Motor Forward
  digitalWrite(motor2pin1, HIGH);
  digitalWrite(motor2pin2, LOW);
}

void turnRight() {
  // Left Motor Forward
  digitalWrite(motor1pin1, HIGH);
  digitalWrite(motor1pin2, LOW);
  // Right Motor Back
  digitalWrite(motor2pin1, LOW);
  digitalWrite(motor2pin2, HIGH);
}

void stopMotors() {
  digitalWrite(motor1pin1, LOW);
  digitalWrite(motor1pin2, LOW);
  digitalWrite(motor2pin1, LOW);
  digitalWrite(motor2pin2, LOW);
}