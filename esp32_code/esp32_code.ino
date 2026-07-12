#include <Arduino.h>
#include <LovyanGFX.hpp>
#include <math.h>

static const bool DEV_MODE = true;

static const int ECU_RX = 16;
static const int ECU_TX = 17;
HardwareSerial ecu(2);

static const int TFT_D0  = 12;
static const int TFT_D1  = 13;
static const int TFT_D2  = 32;
static const int TFT_D3  = 25;
static const int TFT_D4  = 33;
static const int TFT_D5  = 26;
static const int TFT_D6  = 27;
static const int TFT_D7  = 14;

static const int TFT_WR  = 21;
static const int TFT_RS  = 19;
static const int TFT_CS  = 22;
static const int TFT_RST = 23;
static const int TFT_RD  = -1;

class LGFX : public lgfx::LGFX_Device {
  lgfx::Panel_ILI9341 panel;
  lgfx::Bus_Parallel8 bus;

public:
  LGFX() {
    {
      auto cfg = bus.config();
      cfg.freq_write = 20000000;
      cfg.pin_wr = TFT_WR;
      cfg.pin_rd = TFT_RD;
      cfg.pin_rs = TFT_RS;
      cfg.pin_d0 = TFT_D0;
      cfg.pin_d1 = TFT_D1;
      cfg.pin_d2 = TFT_D2;
      cfg.pin_d3 = TFT_D3;
      cfg.pin_d4 = TFT_D4;
      cfg.pin_d5 = TFT_D5;
      cfg.pin_d6 = TFT_D6;
      cfg.pin_d7 = TFT_D7;
      bus.config(cfg);
      panel.setBus(&bus);
    }

    {
      auto cfg = panel.config();
      cfg.pin_cs = TFT_CS;
      cfg.pin_rst = TFT_RST;
      cfg.pin_busy = -1;
      cfg.panel_width = 240;
      cfg.panel_height = 320;
      cfg.memory_width = 240;
      cfg.memory_height = 320;
      panel.config(cfg);
    }

    setPanel(&panel);
  }
};

LGFX tft;

struct EngineData {
  uint16_t rpm;
  float tps;
  float lambda;
  uint8_t duty;
  uint16_t mapKpa;
  int iatC;
  int cltC;
  uint8_t gammaE;
  float battV;
};

EngineData currentData{};
float shownRpm = 0.0f;
float shownTps = 0.0f;
float shownMap = 0.0f;
float shownLambda = 1.0f;

bool readExact(uint8_t *buf, size_t len, uint32_t timeoutMs = 300) {
  uint32_t start = millis();
  size_t got = 0;
  while (got < len && millis() - start < timeoutMs) {
    while (ecu.available() && got < len) {
      buf[got++] = static_cast<uint8_t>(ecu.read());
    }
    delay(1);
  }
  return got == len;
}

uint16_t u16le(const uint8_t *p, int idx) {
  return static_cast<uint16_t>(p[idx]) |
         (static_cast<uint16_t>(p[idx + 1]) << 8);
}

bool requestSpeeduinoData(EngineData &d) {
  while (ecu.available()) ecu.read();
  ecu.write('n');
  ecu.flush();

  uint8_t hdr[3];
  if (!readExact(hdr, 3, 300)) return false;

  uint8_t len = hdr[2];
  uint8_t payload[200];

  if (len > sizeof(payload)) return false;
  if (!readExact(payload, len, 500)) return false;
  if (len < 31) return false;

  const int TEMP_OFFSET = 40;
  d.mapKpa = u16le(payload, 4);
  d.iatC = static_cast<int>(payload[6]) - TEMP_OFFSET;
  d.cltC = static_cast<int>(payload[7]) - TEMP_OFFSET;
  d.battV = payload[9] / 10.0f;
  d.lambda = payload[10] * 0.0110448f;
  d.rpm = u16le(payload, 14);
  d.gammaE = payload[17];
  d.tps = payload[24] / 2.0f;
  d.duty = payload[30];
  return true;
}

float randomFloat(float low, float high) {
  return low + (random(0, 10001) / 10000.0f) * (high - low);
}

float approach(float value, float target, float maxStep) {
  float delta = target - value;
  if (delta > maxStep) delta = maxStep;
  if (delta < -maxStep) delta = -maxStep;
  return value + delta;
}

void generateDevelopmentData(EngineData &d) {
  static float rpm = 900.0f;
  static float tps = 2.0f;
  static float clt = 25.0f;
  static float targetRpm = 900.0f;
  static float targetTps = 2.0f;
  static uint32_t nextChange = 0;

  if (millis() >= nextChange) {
    nextChange = millis() + random(1500, 4500);
    switch (random(0, 5)) {
      case 0: targetRpm = randomFloat(800, 1050); targetTps = randomFloat(1, 4); break;
      case 1: targetRpm = randomFloat(1600, 2600); targetTps = randomFloat(8, 22); break;
      case 2: targetRpm = randomFloat(2500, 4200); targetTps = randomFloat(25, 55); break;
      case 3: targetRpm = randomFloat(4000, 6500); targetTps = randomFloat(60, 100); break;
      default: targetRpm = randomFloat(1200, 3000); targetTps = randomFloat(0, 3); break;
    }
  }

  rpm = approach(rpm, targetRpm, randomFloat(35, 120));
  tps = approach(tps, targetTps, randomFloat(0.8f, 2.8f));

  if (clt < 88) clt += randomFloat(0.03f, 0.10f);
  else clt += randomFloat(-0.15f, 0.15f);

  d.rpm = constrain((int)(rpm + randomFloat(-20, 20)), 0, 7500);
  d.tps = constrain(tps + randomFloat(-0.4f, 0.4f), 0.0f, 100.0f);
  d.mapKpa = constrain((int)(32 + d.tps * 0.70f + randomFloat(-2, 2)), 25, 103);

  if (d.tps > 70) d.lambda = randomFloat(0.82f, 0.90f);
  else if (d.tps < 3 && d.rpm > 1400) d.lambda = randomFloat(1.02f, 1.10f);
  else d.lambda = randomFloat(0.96f, 1.04f);

  d.duty = constrain((int)(3 + d.tps * 0.72f + d.rpm / 470.0f), 0, 95);
  d.iatC = random(25, 39);
  d.cltC = constrain((int)clt, 20, 105);
  d.gammaE = d.cltC < 40 ? random(120, 136) : d.cltC < 70 ? random(105, 121) : random(98, 104);
  d.battV = randomFloat(13.6f, 14.4f);
}

uint16_t rpmColor(float rpm) {
  if (rpm < 4500) return TFT_GREEN;
  if (rpm < 6000) return TFT_YELLOW;
  return TFT_RED;
}

uint16_t lambdaColor(float lambda) {
  if (lambda < 0.90f) return TFT_CYAN;
  if (lambda <= 1.05f) return TFT_GREEN;
  return TFT_RED;
}

uint16_t tempColor(int temperature) {
  if (temperature < 70) return TFT_CYAN;
  if (temperature < 100) return TFT_GREEN;
  if (temperature < 110) return TFT_ORANGE;
  return TFT_RED;
}

void drawStaticLayout() {
  tft.fillScreen(TFT_BLACK);

  tft.drawRoundRect(8, 8, 180, 224, 8, TFT_DARKGREY);
  tft.drawRoundRect(196, 8, 116, 50, 8, TFT_DARKGREY);
  tft.drawRoundRect(196, 64, 116, 50, 8, TFT_DARKGREY);
  tft.drawRoundRect(196, 120, 116, 50, 8, TFT_DARKGREY);
  tft.drawRoundRect(196, 176, 116, 56, 8, TFT_DARKGREY);

  tft.setTextSize(1);
  tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  tft.setCursor(204, 15); tft.print("TPS");
  tft.setCursor(204, 71); tft.print("MAP");
  tft.setCursor(204, 127); tft.print("LAMBDA");
  tft.setCursor(204, 183); tft.print("COOLANT");
}

void drawTachometer(float rpm) {
  const int cx = 98;
  const int cy = 105;
  const int radius = 78;

  tft.fillCircle(cx, cy, radius + 3, TFT_BLACK);

  for (int i = 0; i <= 14; i++) {
    float value = i * 500.0f;
    float angle = (135.0f + (value / 7000.0f) * 270.0f) * DEG_TO_RAD;
    int inner = (i % 2 == 0) ? radius - 14 : radius - 9;
    int x1 = cx + cosf(angle) * inner;
    int y1 = cy + sinf(angle) * inner;
    int x2 = cx + cosf(angle) * radius;
    int y2 = cy + sinf(angle) * radius;
    tft.drawLine(x1, y1, x2, y2, i >= 12 ? TFT_RED : TFT_WHITE);
  }

  float needleAngle = (135.0f + (constrain(rpm, 0.0f, 7000.0f) / 7000.0f) * 270.0f) * DEG_TO_RAD;
  int nx = cx + cosf(needleAngle) * (radius - 20);
  int ny = cy + sinf(needleAngle) * (radius - 20);

  tft.drawLine(cx, cy, nx, ny, rpmColor(rpm));
  tft.drawLine(cx + 1, cy, nx + 1, ny, rpmColor(rpm));
  tft.fillCircle(cx, cy, 6, TFT_WHITE);
  tft.fillCircle(cx, cy, 3, rpmColor(rpm));

  tft.fillRect(35, 135, 126, 50, TFT_BLACK);
  tft.setTextDatum(middle_center);
  tft.setTextColor(rpmColor(rpm), TFT_BLACK);
  tft.setTextSize(4);
  tft.drawString(String((int)rpm), cx, 151);

  tft.setTextSize(1);
  tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  tft.drawString("RPM", cx, 181);
  tft.setTextDatum(top_left);
}

void drawBar(int x, int y, int width, int height, float value, float maximum, uint16_t color) {
  value = constrain(value, 0.0f, maximum);
  int filled = (int)((value / maximum) * (width - 4));

  tft.fillRoundRect(x, y, width, height, 4, TFT_DARKGREY);
  tft.fillRect(x + 2, y + 2, width - 4, height - 4, TFT_BLACK);

  if (filled > 0) {
    tft.fillRect(x + 2, y + 2, filled, height - 4, color);
  }
}

void drawDashboard(const EngineData &d) {
  shownRpm += (d.rpm - shownRpm) * 0.18f;
  shownTps += (d.tps - shownTps) * 0.20f;
  shownMap += (d.mapKpa - shownMap) * 0.20f;
  shownLambda += (d.lambda - shownLambda) * 0.16f;

  drawTachometer(shownRpm);

  tft.fillRect(204, 27, 100, 24, TFT_BLACK);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.setCursor(210, 31);
  tft.printf("%5.1f%%", shownTps);
  drawBar(204, 48, 100, 7, shownTps, 100.0f, shownTps > 75 ? TFT_ORANGE : TFT_CYAN);

  tft.fillRect(204, 82, 100, 25, TFT_BLACK);
  tft.setTextColor(TFT_CYAN, TFT_BLACK);
  tft.setTextSize(2);
  tft.setCursor(210, 86);
  tft.printf("%3d kPa", (int)shownMap);

  tft.fillRect(204, 138, 100, 25, TFT_BLACK);
  tft.setTextColor(lambdaColor(shownLambda), TFT_BLACK);
  tft.setTextSize(2);
  tft.setCursor(218, 142);
  tft.printf("%.2f", shownLambda);

  tft.fillRect(204, 194, 100, 30, TFT_BLACK);
  tft.setTextColor(tempColor(d.cltC), TFT_BLACK);
  tft.setTextSize(2);
  tft.setCursor(220, 199);
  tft.printf("%d C", d.cltC);

  // Compact bottom information row to avoid text overflow.
  tft.fillRect(18, 194, 162, 31, TFT_BLACK);
  tft.setTextSize(1);

  tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  tft.setCursor(20, 196);
  tft.print("DUTY");
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.setCursor(52, 196);
  tft.printf("%u%%", d.duty);

  tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  tft.setCursor(82, 196);
  tft.print("IAT");
  tft.setTextColor(TFT_CYAN, TFT_BLACK);
  tft.setCursor(108, 196);
  tft.printf("%dC", d.iatC);

  tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  tft.setCursor(20, 207);
  tft.print("BATT");
  tft.setTextColor(d.battV < 12.0f ? TFT_RED : TFT_GREEN, TFT_BLACK);
  tft.setCursor(52, 207);
  tft.printf("%.1fV", d.battV);

  drawBar(100, 210, 69, 8, d.duty, 100.0f, d.duty > 85 ? TFT_RED : TFT_GREEN);

  if (d.rpm >= 6500) {
    tft.fillRect(53, 10, 90, 20, TFT_RED);
    tft.setTextColor(TFT_WHITE, TFT_RED);
    tft.setTextSize(2);
    tft.setCursor(69, 12);
    tft.print("SHIFT");
  } else {
    tft.fillRect(53, 10, 90, 20, TFT_BLACK);
  }
}

void setup() {
  Serial.begin(115200);
  randomSeed(esp_random());

  if (!DEV_MODE) {
    ecu.begin(115200, SERIAL_8N1, ECU_RX, ECU_TX);
  }

  tft.init();
  tft.setRotation(1);
  tft.setTextWrap(false);
  drawStaticLayout();
}

void loop() {
  bool valid = DEV_MODE ? (generateDevelopmentData(currentData), true)
                        : requestSpeeduinoData(currentData);

  if (valid) {
    drawDashboard(currentData);
  } else {
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.setTextSize(2);
    tft.setCursor(45, 140);
    tft.print("ECU TIMEOUT");
  }

  delay(80);
}
