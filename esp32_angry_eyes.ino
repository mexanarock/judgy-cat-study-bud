/*
  esp32_angry_eyes.ino
  ---------------------
  Listens on USB serial for "ALERT" / "OK" lines sent by focus_monitor.py
  and drives two independent 1.3" OLED displays (SSD1306-compatible,
  128x64) on two separate I2C buses. Each screen shows ONE eye, so the
  pair together reads as a face: left screen = left eye, right screen =
  right eye.

      Display 1 (bus "Wire",  LEFT eye)  : SDA = GPIO21, SCL = GPIO22
      Display 2 (bus "Wire1", RIGHT eye) : SDA = GPIO14, SCL = GPIO27

  Eye styles:
    CALM  -> a hollow (outline) circle that slowly wanders around the
             screen, like an eye glancing around.
    ANGRY -> just a single thick slanted line (no circle) - the left
             screen's line slants "\" and the right screen's "/", so
             together they form the classic angry brow "V".

  Two servos add physical expression:
      Servo 1 : GPIO2
      Servo 2 : GPIO4
  While ALERT, both servos sweep +/-10 degrees from center (to-and-fro,
  in sync with the eye animation) for a "shaking with anger" look.
  They rest at center (90 degrees) while calm.

  ALERT -> both screens show a single animated angry slash-eye, servos sweep
  OK    -> both screens show a single wandering calm eye, servos rest

  Libraries needed (install via Arduino Library Manager):
    - Adafruit GFX Library
    - Adafruit SSD1306
    - ESP32Servo (by Kevin Harrington / madhephaestus)

  If your 1.3" module actually uses the SH1106 driver instead of SSD1306
  (common for cheap 1.3" boards), swap Adafruit_SSD1306 for the
  "Adafruit_SH110X" library's Adafruit_SH1106G class - the drawing code
  below is unchanged either way.
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>
#include <math.h>

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_ADDR     0x3C   // most modules are 0x3C; some are 0x3D

// ---- Bus 1 pins (LEFT eye) ----
#define BUS1_SDA 21
#define BUS1_SCL 22
// ---- Bus 2 pins (RIGHT eye) ----
#define BUS2_SDA 14
#define BUS2_SCL 27

TwoWire I2Cbus1 = TwoWire(0);
TwoWire I2Cbus2 = TwoWire(1);

Adafruit_SSD1306 display1(SCREEN_WIDTH, SCREEN_HEIGHT, &I2Cbus1, -1); // LEFT eye
Adafruit_SSD1306 display2(SCREEN_WIDTH, SCREEN_HEIGHT, &I2Cbus2, -1); // RIGHT eye

// ---- Expression servos ----
#define SERVO1_PIN 2
#define SERVO2_PIN 4
const int SERVO_CENTER = 90;   // resting angle (degrees)
const int SERVO_SWING  = 10;   // +/- degrees of to-and-fro motion when alerting
Servo servo1;
Servo servo2;

enum Mood { CALM, ANGRY };
Mood currentMood = CALM;

enum EyeSide { EYE_LEFT, EYE_RIGHT };

String serialBuffer = "";
unsigned long lastBlinkToggle = 0;
bool blinkFrame = false;          // alternates for simple animation
const unsigned long BLINK_INTERVAL_MS = 350;

// --------------------------------------------------------------------- //
// Drawing helpers
// --------------------------------------------------------------------- //

// Draws a diagonal bar of given thickness from (x0,y0) to (x1,y1).
// Used for the angry-eye slash line. Works well for the shallow slopes we use.
void drawThickLine(Adafruit_SSD1306 &d, int x0, int y0, int x1, int y1,
                    int thickness, uint16_t color) {
  for (int i = 0; i < thickness; i++) {
    d.drawLine(x0, y0 + i, x1, y1 + i, color);
  }
}

// Single calm eye - a hollow (outline-only) circle that slowly wanders
// around the screen, like an eye glancing around. Both screens use the
// same motion so the two eyes stay "looking" in the same direction.
void drawCalmEye(Adafruit_SSD1306 &d, EyeSide side) {
  d.clearDisplay();

  int baseCX = SCREEN_WIDTH / 2;
  int baseCY = SCREEN_HEIGHT / 2;
  int eyeR = 22;

  // How far the eye is allowed to wander from center, and how fast.
  int wanderRangeX = 20;
  int wanderRangeY = 10;
  float t = millis() / 1000.0f;
  int dx = (int)(wanderRangeX * sinf(t * 0.7f));
  int dy = (int)(wanderRangeY * sinf(t * 0.45f + 1.3f));

  int cx = baseCX + dx;
  int cy = baseCY + dy;

  // Outline only (not filled), drawn 2px thick to stay visible.
  d.drawCircle(cx, cy, eyeR, SSD1306_WHITE);
  d.drawCircle(cx, cy, eyeR - 1, SSD1306_WHITE);

  d.display();
}

// Single angry eye - just a thick slanted line (no eye shape), matching a
// furrowed angry brow. It slants DOWN toward the nose (the eye's inner
// side) and UP toward the temple (outer side); getting the inner/outer
// sides right is what keeps it from reading as upside-down / sad.
void drawAngryEye(Adafruit_SSD1306 &d, EyeSide side) {
  d.clearDisplay();

  int leftX = 26;
  int rightX = SCREEN_WIDTH - 26;
  // Flipped vertically + steeper slope than before: outer(temple) end is
  // now the LOW point, inner(nose) end is the HIGH point.
  int yOuter = 52;  // outer (temple) end - low on screen
  int yInner = 10;  // inner (nose) end - high on screen
  int thickness = 9;

  // Tiny twitch each animation tick for a bit of "shaking with anger" life.
  int wobble = blinkFrame ? 2 : -2;

  if (side == EYE_LEFT) {
    // Outer end is on the left (low), inner end is on the right (high).
    drawThickLine(d, leftX, yOuter + wobble, rightX, yInner + wobble, thickness, SSD1306_WHITE);
  } else {
    // Inner end is on the left (high), outer end is on the right (low).
    drawThickLine(d, leftX, yInner + wobble, rightX, yOuter + wobble, thickness, SSD1306_WHITE);
  }

  d.display();
}

void showMood(Mood m) {
  if (m == ANGRY) {
    drawAngryEye(display1, EYE_LEFT);
    drawAngryEye(display2, EYE_RIGHT);
  } else {
    drawCalmEye(display1, EYE_LEFT);
    drawCalmEye(display2, EYE_RIGHT);
  }
}

// Moves both servos 10 degrees to-and-fro (in sync with the blink toggle)
// while ANGRY, for a bit of physical "shaking with anger" expression.
// Rests at center when CALM.
void updateServos(Mood m, bool toggle) {
  if (m == ANGRY) {
    int angle = toggle ? (SERVO_CENTER + SERVO_SWING) : (SERVO_CENTER - SERVO_SWING);
    servo1.write(angle);
    servo2.write(angle);
  } else {
    servo1.write(SERVO_CENTER);
    servo2.write(SERVO_CENTER);
  }
}

// --------------------------------------------------------------------- //
// Setup / Loop
// --------------------------------------------------------------------- //
void setup() {
  Serial.begin(115200);
  delay(200);

  I2Cbus1.begin(BUS1_SDA, BUS1_SCL, 400000);
  I2Cbus2.begin(BUS2_SDA, BUS2_SCL, 400000);

  if (!display1.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("display1 init failed");
  }
  if (!display2.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("display2 init failed");
  }

  // Physical displays are mounted upside down -> rotate output 180°.
  // Valid values: 0 = normal, 1 = 90°, 2 = 180°, 3 = 270°.
  display1.setRotation(2);
  display2.setRotation(2);

  display1.clearDisplay();
  display2.clearDisplay();
  display1.display();
  display2.display();

  // ---- servos ----
  // ESP32Servo needs the LEDC PWM timers allocated before attach().
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  servo1.setPeriodHertz(50);   // standard 50Hz servo
  servo2.setPeriodHertz(50);
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);
  servo1.write(SERVO_CENTER);
  servo2.write(SERVO_CENTER);

  showMood(CALM);
  Serial.println("ESP32 angry-eyes controller ready");
}

void loop() {
  // ---- read serial commands ----
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      serialBuffer.trim();
      if (serialBuffer.equalsIgnoreCase("ALERT")) {
        currentMood = ANGRY;
      } else if (serialBuffer.equalsIgnoreCase("OK")) {
        currentMood = CALM;
      }
      serialBuffer = "";
    } else if (c != '\r') {
      serialBuffer += c;
    }
  }

  // ---- simple blink/squint animation, redraw periodically ----
  unsigned long nowMs = millis();
  if (nowMs - lastBlinkToggle >= BLINK_INTERVAL_MS) {
    lastBlinkToggle = nowMs;
    blinkFrame = !blinkFrame;
    showMood(currentMood);
    updateServos(currentMood, blinkFrame);
  }
}
