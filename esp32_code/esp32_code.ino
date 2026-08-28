#include <Arduino.h>
#include <LovyanGFX.hpp>
#include <LittleFS.h>
#include <AnimatedGIF.h>
#include <math.h>

// =========================================================
// GENERAL
// =========================================================

static const bool DEV_MODE = false;

// =========================================================
// ECU SERIAL
// =========================================================

static const int ECU_RX = 16;
static const int ECU_TX = 17;
HardwareSerial ecu(2);

// =========================================================
// BUTTON
// GPIO15 ---- BUTTON ---- GND
// =========================================================

static const int BUTTON_PIN = 15;
static const uint32_t BUTTON_DEBOUNCE_MS = 30;
static const uint32_t BUTTON_LONG_MS = 900;

bool lastButtonReading = HIGH;
bool stableButtonState = HIGH;
uint32_t buttonChangedAt = 0;
uint32_t buttonPressedAt = 0;
bool longPressTriggered = false;

// =========================================================
// TFT PINS
// =========================================================

static const int TFT_D0 = 12;
static const int TFT_D1 = 13;
static const int TFT_D2 = 32;
static const int TFT_D3 = 33;
static const int TFT_D4 = 25;
static const int TFT_D5 = 26;
static const int TFT_D6 = 27;
static const int TFT_D7 = 14;

static const int TFT_WR = 19;
static const int TFT_RS = 21;
static const int TFT_CS = 22;
static const int TFT_RST = 23;
static const int TFT_RD = 18;

// =========================================================
// DISPLAY
// =========================================================

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

// Used only for non-GIF screens.
// This keeps dashboard/RPM/telemetry/bars flicker-free.
LGFX_Sprite uiSprite(&tft);
bool uiSpriteReady = false;

// =========================================================
// ENGINE DATA
// =========================================================

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

// =========================================================
// SMOOTH VALUES
// =========================================================

float shownRpm = 0.0f;
float shownTps = 0.0f;
float shownMap = 0.0f;
float shownLambda = 1.0f;

// =========================================================
// SCREENS
// =========================================================

enum ScreenID {
  SCREEN_DASHBOARD = 0,
  SCREEN_RPM,
  SCREEN_TELEMETRY,
  SCREEN_BARS,
  SCREEN_SAMURAI,
  SCREEN_COUNT
};

enum ButtonEvent {
  BUTTON_NONE,
  BUTTON_SHORT,
  BUTTON_LONG
};

ScreenID currentScreen = SCREEN_DASHBOARD;
bool screenNeedsRedraw = true;

// =========================================================
// GIF
// data/giphy.gif -> LittleFS path /giphy.gif
//
// THIS IS THE SAME WORKING DIRECT-TO-TFT APPROACH.
// =========================================================

AnimatedGIF gif;

static const char *SAMURAI_GIF_PATH = "/giphy.gif";

bool gifOpened = false;
bool littleFSReady = false;

int gifXOffset = 0;
int gifYOffset = 24;

uint32_t nextGifFrameAt = 0;

static const int GIF_AREA_TOP = 24;
static const int GIF_AREA_BOTTOM = 187;
static const int GIF_AREA_HEIGHT =
  GIF_AREA_BOTTOM - GIF_AREA_TOP + 1;

static uint16_t gifLineBuffer[480];

// =========================================================
// HELPERS
// =========================================================

float randomFloat(float low, float high) {
  return low +
         (random(0, 10001) / 10000.0f) *
         (high - low);
}

float approach(float value, float target, float maxStep) {
  float delta = target - value;

  if (delta > maxStep) delta = maxStep;
  if (delta < -maxStep) delta = -maxStep;

  return value + delta;
}

uint16_t u16le(const uint8_t *p, int idx) {
  return static_cast<uint16_t>(p[idx]) |
         (static_cast<uint16_t>(p[idx + 1]) << 8);
}

// =========================================================
// COLORS
// =========================================================

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

// =========================================================
// SPEEDUINO SERIAL
// =========================================================

bool readExact(
  uint8_t *buf,
  size_t len,
  uint32_t timeoutMs = 300
) {
  uint32_t start = millis();
  size_t got = 0;

  while (
    got < len &&
    millis() - start < timeoutMs
  ) {
    while (
      ecu.available() &&
      got < len
    ) {
      buf[got++] =
        static_cast<uint8_t>(
          ecu.read()
        );
    }

    delay(1);
  }

  return got == len;
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

// =========================================================
// DEVELOPMENT DATA
// =========================================================

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
      case 0:
        targetRpm = randomFloat(800, 1050);
        targetTps = randomFloat(1, 4);
        break;

      case 1:
        targetRpm = randomFloat(1600, 2600);
        targetTps = randomFloat(8, 22);
        break;

      case 2:
        targetRpm = randomFloat(2500, 4200);
        targetTps = randomFloat(25, 55);
        break;

      case 3:
        targetRpm = randomFloat(4000, 6800);
        targetTps = randomFloat(60, 100);
        break;

      default:
        targetRpm = randomFloat(1200, 3000);
        targetTps = randomFloat(0, 3);
        break;
    }
  }

  rpm =
    approach(
      rpm,
      targetRpm,
      randomFloat(35, 120)
    );

  tps =
    approach(
      tps,
      targetTps,
      randomFloat(0.8f, 2.8f)
    );

  if (clt < 88) {
    clt += randomFloat(0.03f, 0.10f);
  } else {
    clt += randomFloat(-0.15f, 0.15f);
  }

  d.rpm =
    constrain(
      (int)(
        rpm +
        randomFloat(-20, 20)
      ),
      0,
      7500
    );

  d.tps =
    constrain(
      tps +
      randomFloat(-0.4f, 0.4f),
      0.0f,
      100.0f
    );

  d.mapKpa =
    constrain(
      (int)(
        32 +
        d.tps * 0.70f +
        randomFloat(-2, 2)
      ),
      25,
      103
    );

  if (d.tps > 70) {
    d.lambda = randomFloat(0.82f, 0.90f);
  } else if (
    d.tps < 3 &&
    d.rpm > 1400
  ) {
    d.lambda = randomFloat(1.02f, 1.10f);
  } else {
    d.lambda = randomFloat(0.96f, 1.04f);
  }

  d.duty =
    constrain(
      (int)(
        3 +
        d.tps * 0.72f +
        d.rpm / 470.0f
      ),
      0,
      95
    );

  d.iatC = random(25, 39);
  d.cltC = constrain((int)clt, 20, 105);

  d.gammaE =
    d.cltC < 40
      ? random(120, 136)
      : d.cltC < 70
        ? random(105, 121)
        : random(98, 104);

  d.battV =
    randomFloat(
      13.6f,
      14.4f
    );
}

// =========================================================
// GIF FILE CALLBACKS
// EXACTLY THE WORKING STYLE
// =========================================================

void *GIFOpenFile(const char *fname, int32_t *pSize) {
  File *f = new File(LittleFS.open(fname, "r"));

  if (!(*f)) {
    Serial.printf("GIF open failed: %s\n", fname);
    delete f;
    return nullptr;
  }

  *pSize = f->size();
  return static_cast<void *>(f);
}

void GIFCloseFile(void *pHandle) {
  File *f = static_cast<File *>(pHandle);

  if (f) {
    f->close();
    delete f;
  }
}

int32_t GIFReadFile(
  GIFFILE *pFile,
  uint8_t *pBuf,
  int32_t iLen
) {
  File *f =
    static_cast<File *>(
      pFile->fHandle
    );

  if (!f) return 0;

  int32_t remaining =
    pFile->iSize -
    pFile->iPos;

  if (iLen > remaining) {
    iLen = remaining;
  }

  int32_t bytesRead =
    f->read(
      pBuf,
      iLen
    );

  pFile->iPos +=
    bytesRead;

  return bytesRead;
}

int32_t GIFSeekFile(
  GIFFILE *pFile,
  int32_t iPosition
) {
  File *f =
    static_cast<File *>(
      pFile->fHandle
    );

  if (!f) return 0;

  if (!f->seek(iPosition)) {
    return pFile->iPos;
  }

  pFile->iPos =
    iPosition;

  return iPosition;
}

// =========================================================
// GIF DRAW CALLBACK
// EXACTLY THE DIRECT-TO-TFT METHOD THAT WORKED
// =========================================================

void GIFDraw(GIFDRAW *pDraw) {
  uint8_t *src =
    pDraw->pPixels;

  uint16_t *palette =
    pDraw->pPalette;

  int dstY =
    gifYOffset +
    pDraw->iY +
    pDraw->y;

  if (
    dstY < GIF_AREA_TOP ||
    dstY > GIF_AREA_BOTTOM
  ) {
    return;
  }

  int frameX =
    gifXOffset +
    pDraw->iX;

  const int width =
    pDraw->iWidth;

  if (!pDraw->ucHasTransparency) {
    int srcStart = 0;
    int drawX = frameX;
    int drawW = width;

    if (drawX < 0) {
      srcStart = -drawX;
      drawW -= srcStart;
      drawX = 0;
    }

    if (drawX + drawW > 320) {
      drawW = 320 - drawX;
    }

    if (drawW <= 0) return;

    for (
      int i = 0;
      i < drawW;
      i++
    ) {
      gifLineBuffer[i] =
        palette[
          src[srcStart + i]
        ];
    }

    tft.pushImage(
      drawX,
      dstY,
      drawW,
      1,
      gifLineBuffer
    );

    return;
  }

  const uint8_t transparentIndex =
    pDraw->ucTransparent;

  int x = 0;

  while (x < width) {
    while (
      x < width &&
      src[x] ==
        transparentIndex
    ) {
      x++;
    }

    if (x >= width) break;

    int runStart = x;

    while (
      x < width &&
      src[x] !=
        transparentIndex
    ) {
      x++;
    }

    int runLength =
      x - runStart;

    int drawX =
      frameX +
      runStart;

    int srcOffset =
      runStart;

    int drawW =
      runLength;

    if (drawX < 0) {
      int clip = -drawX;
      srcOffset += clip;
      drawW -= clip;
      drawX = 0;
    }

    if (drawX + drawW > 320) {
      drawW = 320 - drawX;
    }

    if (drawW <= 0) continue;

    for (
      int i = 0;
      i < drawW;
      i++
    ) {
      gifLineBuffer[i] =
        palette[
          src[srcOffset + i]
        ];
    }

    tft.pushImage(
      drawX,
      dstY,
      drawW,
      1,
      gifLineBuffer
    );
  }
}

// =========================================================
// GIF MANAGEMENT
// SAME WORKING API
// =========================================================

void stopSamuraiGif() {
  if (gifOpened) {
    gif.close();
    gifOpened = false;
  }

  nextGifFrameAt = 0;
}

bool startSamuraiGif() {
  stopSamuraiGif();

  if (!littleFSReady) {
    Serial.println(
      "LittleFS is not ready"
    );
    return false;
  }

  if (
    !LittleFS.exists(
      SAMURAI_GIF_PATH
    )
  ) {
    Serial.println(
      "ERROR: /giphy.gif not found in LittleFS"
    );
    return false;
  }

  if (
    !gif.open(
      SAMURAI_GIF_PATH,
      GIFOpenFile,
      GIFCloseFile,
      GIFReadFile,
      GIFSeekFile,
      GIFDraw
    )
  ) {
    Serial.printf(
      "AnimatedGIF open error: %d\n",
      gif.getLastError()
    );

    return false;
  }

  int gifW =
    gif.getCanvasWidth();

  int gifH =
    gif.getCanvasHeight();

  gifXOffset =
    (320 - gifW) / 2;

  gifYOffset =
    GIF_AREA_TOP +
    (
      GIF_AREA_HEIGHT -
      gifH
    ) / 2;

  if (gifXOffset < 0) {
    gifXOffset = 0;
  }

  if (
    gifYOffset <
    GIF_AREA_TOP
  ) {
    gifYOffset =
      GIF_AREA_TOP;
  }

  Serial.printf(
    "GIF opened: %dx%d, offset=(%d,%d)\n",
    gifW,
    gifH,
    gifXOffset,
    gifYOffset
  );

  gifOpened = true;
  nextGifFrameAt = 0;

  return true;
}

void updateSamuraiGif() {
  if (!gifOpened) {
    return;
  }

  uint32_t now =
    millis();

  if (
    now <
    nextGifFrameAt
  ) {
    return;
  }

  int frameDelayMs = 0;

  int result =
    gif.playFrame(
      false,
      &frameDelayMs
    );

  if (!result) {
    gif.reset();

    result =
      gif.playFrame(
        false,
        &frameDelayMs
      );
  }

  if (
    frameDelayMs <
    20
  ) {
    frameDelayMs = 20;
  }

  nextGifFrameAt =
    millis() +
    frameDelayMs;
}

// =========================================================
// BUTTON
// =========================================================

ButtonEvent updateButton() {
  bool reading =
    digitalRead(
      BUTTON_PIN
    );

  if (
    reading !=
    lastButtonReading
  ) {
    lastButtonReading =
      reading;

    buttonChangedAt =
      millis();
  }

  if (
    millis() -
    buttonChangedAt >
    BUTTON_DEBOUNCE_MS
  ) {
    if (
      reading !=
      stableButtonState
    ) {
      stableButtonState =
        reading;

      if (
        stableButtonState ==
        LOW
      ) {
        buttonPressedAt =
          millis();

        longPressTriggered =
          false;
      } else {
        if (!longPressTriggered) {
          return BUTTON_SHORT;
        }
      }
    }
  }

  if (
    stableButtonState == LOW &&
    !longPressTriggered &&
    millis() -
      buttonPressedAt >=
      BUTTON_LONG_MS
  ) {
    longPressTriggered = true;
    return BUTTON_LONG;
  }

  return BUTTON_NONE;
}

void changeScreen(
  ScreenID newScreen
) {
  if (
    currentScreen ==
      SCREEN_SAMURAI &&
    newScreen !=
      SCREEN_SAMURAI
  ) {
    stopSamuraiGif();
  }

  currentScreen =
    newScreen;

  screenNeedsRedraw =
    true;
}

void handleButton() {
  ButtonEvent event =
    updateButton();

  if (
    event ==
    BUTTON_SHORT
  ) {
    ScreenID nextScreen =
      static_cast<ScreenID>(
        (
          currentScreen +
          1
        ) %
        SCREEN_COUNT
      );

    changeScreen(
      nextScreen
    );
  }

  if (
    event ==
    BUTTON_LONG
  ) {
    changeScreen(
      SCREEN_DASHBOARD
    );
  }
}

// =========================================================
// SPRITE BAR
// =========================================================

void drawBar(
  LGFX_Sprite &g,
  int x,
  int y,
  int width,
  int height,
  float value,
  float maximum,
  uint16_t color
) {
  value =
    constrain(
      value,
      0.0f,
      maximum
    );

  int filled =
    (int)(
      (
        value /
        maximum
      ) *
      (width - 4)
    );

  g.fillRoundRect(
    x,
    y,
    width,
    height,
    4,
    TFT_DARKGREY
  );

  g.fillRect(
    x + 2,
    y + 2,
    width - 4,
    height - 4,
    TFT_BLACK
  );

  if (filled > 0) {
    g.fillRect(
      x + 2,
      y + 2,
      filled,
      height - 4,
      color
    );
  }
}

// =========================================================
// DASHBOARD
// =========================================================

void drawDashboardTachometer(
  LGFX_Sprite &g,
  float rpm
) {
  const int cx = 98;
  const int cy = 105;
  const int radius = 78;

  for (
    int i = 0;
    i <= 14;
    i++
  ) {
    float value =
      i * 500.0f;

    float angle =
      (
        135.0f +
        (
          value /
          7000.0f
        ) *
        270.0f
      ) *
      DEG_TO_RAD;

    int inner =
      (
        i % 2 == 0
      )
        ? radius - 14
        : radius - 9;

    int x1 =
      cx +
      cosf(angle) *
      inner;

    int y1 =
      cy +
      sinf(angle) *
      inner;

    int x2 =
      cx +
      cosf(angle) *
      radius;

    int y2 =
      cy +
      sinf(angle) *
      radius;

    uint16_t tickColor =
      value >= 6000
        ? TFT_RED
        : TFT_WHITE;

    g.drawLine(
      x1,
      y1,
      x2,
      y2,
      tickColor
    );
  }

  float needleAngle =
    (
      135.0f +
      (
        constrain(
          rpm,
          0.0f,
          7000.0f
        ) /
        7000.0f
      ) *
      270.0f
    ) *
    DEG_TO_RAD;

  int nx =
    cx +
    cosf(
      needleAngle
    ) *
    (radius - 20);

  int ny =
    cy +
    sinf(
      needleAngle
    ) *
    (radius - 20);

  uint16_t col =
    rpmColor(rpm);

  g.drawLine(
    cx,
    cy,
    nx,
    ny,
    col
  );

  g.drawLine(
    cx + 1,
    cy,
    nx + 1,
    ny,
    col
  );

  g.fillCircle(
    cx,
    cy,
    6,
    TFT_WHITE
  );

  g.fillCircle(
    cx,
    cy,
    3,
    col
  );

  g.setTextDatum(
    middle_center
  );

  g.setTextColor(
    col,
    TFT_BLACK
  );

  g.setTextSize(4);

  g.drawString(
    String((int)rpm),
    cx,
    151
  );

  g.setTextSize(1);

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.drawString(
    "RPM",
    cx,
    181
  );

  g.setTextDatum(
    top_left
  );
}

void drawDashboard(
  const EngineData &d
) {
  if (!uiSpriteReady) return;

  shownRpm +=
    (
      d.rpm -
      shownRpm
    ) *
    0.18f;

  shownTps +=
    (
      d.tps -
      shownTps
    ) *
    0.20f;

  shownMap +=
    (
      d.mapKpa -
      shownMap
    ) *
    0.20f;

  shownLambda +=
    (
      d.lambda -
      shownLambda
    ) *
    0.16f;

  LGFX_Sprite &g =
    uiSprite;

  g.fillSprite(
    TFT_BLACK
  );

  g.drawRoundRect(
    8,
    8,
    180,
    224,
    8,
    TFT_DARKGREY
  );

  g.drawRoundRect(
    196,
    8,
    116,
    50,
    8,
    TFT_DARKGREY
  );

  g.drawRoundRect(
    196,
    64,
    116,
    50,
    8,
    TFT_DARKGREY
  );

  g.drawRoundRect(
    196,
    120,
    116,
    50,
    8,
    TFT_DARKGREY
  );

  g.drawRoundRect(
    196,
    176,
    116,
    56,
    8,
    TFT_DARKGREY
  );

  g.setTextSize(1);

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.setCursor(204, 15);
  g.print("TPS");

  g.setCursor(204, 71);
  g.print("MAP");

  g.setCursor(204, 127);
  g.print("LAMBDA");

  g.setCursor(204, 183);
  g.print("COOLANT");

  drawDashboardTachometer(
    g,
    shownRpm
  );

  g.setTextColor(
    TFT_WHITE,
    TFT_BLACK
  );

  g.setTextSize(2);
  g.setCursor(210, 31);

  g.printf(
    "%5.1f%%",
    shownTps
  );

  drawBar(
    g,
    204,
    48,
    100,
    7,
    shownTps,
    100.0f,
    shownTps > 75
      ? TFT_ORANGE
      : TFT_CYAN
  );

  g.setTextColor(
    TFT_CYAN,
    TFT_BLACK
  );

  g.setCursor(210, 86);

  g.printf(
    "%3d kPa",
    (int)shownMap
  );

  g.setTextColor(
    lambdaColor(
      shownLambda
    ),
    TFT_BLACK
  );

  g.setCursor(218, 142);

  g.printf(
    "%.2f",
    shownLambda
  );

  g.setTextColor(
    tempColor(d.cltC),
    TFT_BLACK
  );

  g.setCursor(220, 199);

  g.printf(
    "%d C",
    d.cltC
  );

  g.setTextSize(1);

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.setCursor(20, 196);
  g.print("DUTY");

  g.setTextColor(
    TFT_GREEN,
    TFT_BLACK
  );

  g.setCursor(52, 196);
  g.printf("%u%%", d.duty);

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.setCursor(82, 196);
  g.print("IAT");

  g.setTextColor(
    TFT_CYAN,
    TFT_BLACK
  );

  g.setCursor(108, 196);
  g.printf("%dC", d.iatC);

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.setCursor(20, 207);
  g.print("BATT");

  g.setTextColor(
    d.battV < 12.0f
      ? TFT_RED
      : TFT_GREEN,
    TFT_BLACK
  );

  g.setCursor(52, 207);
  g.printf("%.1fV", d.battV);

  drawBar(
    g,
    100,
    210,
    69,
    8,
    d.duty,
    100.0f,
    d.duty > 85
      ? TFT_RED
      : TFT_GREEN
  );

  if (
    d.rpm >= 6500 &&
    (
      (
        millis() /
        100
      ) %
      2
    )
  ) {
    g.fillRect(
      53,
      10,
      90,
      20,
      TFT_RED
    );

    g.setTextColor(
      TFT_WHITE,
      TFT_RED
    );

    g.setTextSize(2);
    g.setCursor(69, 12);
    g.print("SHIFT");
  }

  g.pushSprite(0, 0);
}

// =========================================================
// RPM ANALOG SCREEN
// =========================================================

void drawAnalogTachometer(
  LGFX_Sprite &g,
  float rpm
) {
  const int cx = 160;
  const int cy = 127;
  const int radius = 101;

  const float startAngle =
    140.0f;

  const float sweepAngle =
    260.0f;

  g.fillSprite(
    TFT_BLACK
  );

  g.setTextDatum(
    middle_center
  );

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.setTextSize(1);

  g.drawString(
    "ENGINE SPEED",
    160,
    11
  );

  g.drawCircle(
    cx,
    cy,
    radius,
    TFT_DARKGREY
  );

  g.drawCircle(
    cx,
    cy,
    radius - 2,
    TFT_DARKGREY
  );

  for (
    int rpmTick = 0;
    rpmTick <= 7000;
    rpmTick += 250
  ) {
    float ratio =
      rpmTick /
      7000.0f;

    float angle =
      (
        startAngle +
        ratio *
        sweepAngle
      ) *
      DEG_TO_RAD;

    bool major =
      rpmTick %
      1000 == 0;

    bool medium =
      rpmTick %
      500 == 0;

    int outerR =
      radius - 5;

    int innerR =
      major
        ? radius - 23
        : medium
          ? radius - 17
          : radius - 11;

    int x1 =
      cx +
      cosf(angle) *
      innerR;

    int y1 =
      cy +
      sinf(angle) *
      innerR;

    int x2 =
      cx +
      cosf(angle) *
      outerR;

    int y2 =
      cy +
      sinf(angle) *
      outerR;

    uint16_t tickColor =
      rpmTick >= 6000
        ? TFT_RED
        : rpmTick >= 5000
          ? TFT_YELLOW
          : TFT_WHITE;

    g.drawLine(
      x1,
      y1,
      x2,
      y2,
      tickColor
    );

    if (major) {
      int textR =
        radius - 35;

      int tx =
        cx +
        cosf(angle) *
        textR;

      int ty =
        cy +
        sinf(angle) *
        textR;

      g.setTextDatum(
        middle_center
      );

      g.setTextSize(1);

      g.setTextColor(
        rpmTick >= 6000
          ? TFT_RED
          : TFT_LIGHTGREY,
        TFT_BLACK
      );

      g.drawString(
        String(
          rpmTick /
          1000
        ),
        tx,
        ty
      );
    }
  }

  for (
    int rpmArc = 6000;
    rpmArc <= 7000;
    rpmArc += 60
  ) {
    float ratio =
      rpmArc /
      7000.0f;

    float angle =
      (
        startAngle +
        ratio *
        sweepAngle
      ) *
      DEG_TO_RAD;

    int x =
      cx +
      cosf(angle) *
      (radius - 10);

    int y =
      cy +
      sinf(angle) *
      (radius - 10);

    g.fillCircle(
      x,
      y,
      2,
      TFT_RED
    );
  }

  float limitedRPM =
    constrain(
      rpm,
      0.0f,
      7000.0f
    );

  float needleAngle =
    (
      startAngle +
      (
        limitedRPM /
        7000.0f
      ) *
      sweepAngle
    ) *
    DEG_TO_RAD;

  int needleLength =
    radius - 30;

  int nx =
    cx +
    cosf(
      needleAngle
    ) *
    needleLength;

  int ny =
    cy +
    sinf(
      needleAngle
    ) *
    needleLength;

  g.drawLine(
    cx + 2,
    cy + 2,
    nx + 2,
    ny + 2,
    TFT_DARKGREY
  );

  g.drawLine(
    cx,
    cy,
    nx,
    ny,
    TFT_RED
  );

  g.drawLine(
    cx + 1,
    cy,
    nx + 1,
    ny,
    TFT_RED
  );

  g.drawLine(
    cx,
    cy + 1,
    nx,
    ny + 1,
    TFT_RED
  );

  int bx =
    cx -
    cosf(
      needleAngle
    ) *
    18;

  int by =
    cy -
    sinf(
      needleAngle
    ) *
    18;

  g.drawLine(
    cx,
    cy,
    bx,
    by,
    TFT_RED
  );

  g.fillCircle(
    cx,
    cy,
    9,
    TFT_DARKGREY
  );

  g.fillCircle(
    cx,
    cy,
    6,
    TFT_RED
  );

  g.fillCircle(
    cx,
    cy,
    2,
    TFT_WHITE
  );

  g.fillRoundRect(
    105,
    158,
    110,
    35,
    5,
    TFT_BLACK
  );

  g.drawRoundRect(
    105,
    158,
    110,
    35,
    5,
    TFT_DARKGREY
  );

  g.setTextDatum(
    middle_center
  );

  g.setTextColor(
    rpmColor(rpm),
    TFT_BLACK
  );

  g.setTextSize(2);

  g.drawString(
    String((int)rpm),
    160,
    170
  );

  g.setTextSize(1);

  g.setTextColor(
    TFT_DARKGREY,
    TFT_BLACK
  );

  g.drawString(
    "RPM",
    160,
    187
  );

  if (
    rpm >= 6500 &&
    (
      (
        millis() /
        100
      ) %
      2
    )
  ) {
    g.fillRoundRect(
      115,
      4,
      90,
      23,
      5,
      TFT_RED
    );

    g.setTextColor(
      TFT_WHITE,
      TFT_RED
    );

    g.setTextSize(2);

    g.drawString(
      "SHIFT",
      160,
      15
    );
  }

  g.setTextDatum(
    top_left
  );
}

void drawRPMScreen(
  const EngineData &d
) {
  if (!uiSpriteReady) return;

  static float displayRPM =
    800.0f;

  displayRPM +=
    (
      d.rpm -
      displayRPM
    ) *
    0.20f;

  drawAnalogTachometer(
    uiSprite,
    displayRPM
  );

  uiSprite.pushSprite(
    0,
    0
  );
}

// =========================================================
// TELEMETRY
// =========================================================

void drawTelemetry(
  const EngineData &d
) {
  if (!uiSpriteReady) return;

  LGFX_Sprite &g =
    uiSprite;

  g.fillSprite(
    TFT_BLACK
  );

  g.setTextColor(
    TFT_WHITE,
    TFT_BLACK
  );

  g.setTextSize(2);
  g.setCursor(10, 8);
  g.print("TELEMETRY");

  g.drawFastHLine(
    10,
    30,
    300,
    TFT_DARKGREY
  );

  g.setTextSize(1);

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.setCursor(15, 48);
  g.print("RPM");

  g.setCursor(15, 90);
  g.print("TPS");

  g.setCursor(15, 132);
  g.print("MAP");

  g.setCursor(15, 174);
  g.print("LAMBDA");

  g.setCursor(170, 48);
  g.print("COOLANT");

  g.setCursor(170, 90);
  g.print("IAT");

  g.setCursor(170, 132);
  g.print("BATTERY");

  g.setCursor(170, 174);
  g.print("DUTY");

  g.setTextSize(2);

  g.setTextColor(
    rpmColor(d.rpm),
    TFT_BLACK
  );

  g.setCursor(15, 62);
  g.printf("%u", d.rpm);

  g.setTextColor(
    TFT_CYAN,
    TFT_BLACK
  );

  g.setCursor(15, 104);
  g.printf("%.1f%%", d.tps);

  g.setCursor(15, 146);
  g.printf("%u kPa", d.mapKpa);

  g.setTextColor(
    lambdaColor(d.lambda),
    TFT_BLACK
  );

  g.setCursor(15, 188);
  g.printf("%.2f", d.lambda);

  g.setTextColor(
    tempColor(d.cltC),
    TFT_BLACK
  );

  g.setCursor(170, 62);
  g.printf("%d C", d.cltC);

  g.setTextColor(
    TFT_CYAN,
    TFT_BLACK
  );

  g.setCursor(170, 104);
  g.printf("%d C", d.iatC);

  g.setTextColor(
    d.battV < 12.0f
      ? TFT_RED
      : TFT_GREEN,
    TFT_BLACK
  );

  g.setCursor(170, 146);
  g.printf("%.1f V", d.battV);

  g.setTextColor(
    TFT_GREEN,
    TFT_BLACK
  );

  g.setCursor(170, 188);
  g.printf("%u%%", d.duty);

  g.pushSprite(
    0,
    0
  );
}

// =========================================================
// BARS
// =========================================================

void drawBarsScreen(
  const EngineData &d
) {
  if (!uiSpriteReady) return;

  LGFX_Sprite &g =
    uiSprite;

  g.fillSprite(
    TFT_BLACK
  );

  g.setTextColor(
    TFT_WHITE,
    TFT_BLACK
  );

  g.setTextSize(2);
  g.setCursor(10, 8);
  g.print("ENGINE LOAD");

  g.drawFastHLine(
    10,
    30,
    300,
    TFT_DARKGREY
  );

  g.setTextSize(1);

  g.setTextColor(
    TFT_LIGHTGREY,
    TFT_BLACK
  );

  g.setCursor(10, 48);
  g.print("RPM");

  g.setCursor(10, 85);
  g.print("THROTTLE");

  g.setCursor(10, 122);
  g.print("MAP");

  g.setCursor(10, 159);
  g.print("INJECTOR");

  g.setCursor(10, 196);
  g.print("COOLANT");

  drawBar(
    g,
    10,
    60,
    220,
    15,
    d.rpm,
    7000,
    rpmColor(d.rpm)
  );

  drawBar(
    g,
    10,
    97,
    220,
    15,
    d.tps,
    100,
    TFT_CYAN
  );

  drawBar(
    g,
    10,
    134,
    220,
    15,
    d.mapKpa,
    110,
    TFT_GREEN
  );

  drawBar(
    g,
    10,
    171,
    220,
    15,
    d.duty,
    100,
    d.duty > 85
      ? TFT_RED
      : TFT_GREEN
  );

  drawBar(
    g,
    10,
    208,
    220,
    15,
    d.cltC,
    120,
    tempColor(d.cltC)
  );

  g.setTextDatum(
    middle_center
  );

  g.setTextSize(2);

  g.setTextColor(
    rpmColor(d.rpm),
    TFT_BLACK
  );

  g.drawString(
    String(d.rpm),
    278,
    67
  );

  g.setTextColor(
    TFT_CYAN,
    TFT_BLACK
  );

  g.drawString(
    String(d.tps, 0) + "%",
    278,
    104
  );

  g.drawString(
    String(d.mapKpa),
    278,
    141
  );

  g.setTextColor(
    TFT_GREEN,
    TFT_BLACK
  );

  g.drawString(
    String(d.duty) + "%",
    278,
    178
  );

  g.setTextColor(
    tempColor(d.cltC),
    TFT_BLACK
  );

  g.drawString(
    String(d.cltC) + "C",
    278,
    215
  );

  g.setTextDatum(
    top_left
  );

  g.pushSprite(
    0,
    0
  );
}

// =========================================================
// SAMURAI GIF SCREEN
// KEEPING THE OLD WORKING METHOD
// =========================================================

void drawSamuraiStatic() {
  tft.fillScreen(
    TFT_BLACK
  );

  tft.setTextDatum(
    middle_center
  );

  tft.setTextColor(
    tft.color565(
      110,
      110,
      115
    ),
    TFT_BLACK
  );

  tft.setTextSize(1);

  tft.drawString(
    "SAMURAI // ENGINE",
    160,
    11
  );

  tft.drawFastHLine(
    18,
    21,
    284,
    tft.color565(
      45,
      45,
      50
    )
  );

  tft.drawFastHLine(
    18,
    191,
    284,
    tft.color565(
      45,
      45,
      50
    )
  );

  tft.setTextDatum(
    top_left
  );

  if (!startSamuraiGif()) {
    tft.setTextDatum(
      middle_center
    );

    tft.setTextColor(
      TFT_RED,
      TFT_BLACK
    );

    tft.setTextSize(2);

    tft.drawString(
      "GIF NOT FOUND",
      160,
      95
    );

    tft.setTextSize(1);

    tft.setTextColor(
      TFT_LIGHTGREY,
      TFT_BLACK
    );

    tft.drawString(
      "/giphy.gif",
      160,
      118
    );

    tft.setTextDatum(
      top_left
    );
  }
}

void drawSamuraiScreen(
  const EngineData &d
) {
  updateSamuraiGif();

  // Only redraw the HUD below the GIF.
  // The GIF itself is still drawn directly by GIFDraw().
  tft.fillRect(
    0,
    193,
    320,
    47,
    TFT_BLACK
  );

  tft.drawFastHLine(
    18,
    194,
    284,
    tft.color565(
      45,
      45,
      50
    )
  );

  tft.setTextDatum(
    middle_center
  );

  tft.setTextColor(
    rpmColor(d.rpm),
    TFT_BLACK
  );

  tft.setTextSize(2);

  tft.drawString(
    String(d.rpm),
    160,
    209
  );

  tft.setTextSize(1);

  tft.setTextColor(
    tft.color565(
      95,
      95,
      100
    ),
    TFT_BLACK
  );

  tft.drawString(
    "RPM",
    160,
    225
  );

  float ratio =
    constrain(
      d.rpm /
      7000.0f,
      0.0f,
      1.0f
    );

  int barW =
    (int)(
      ratio *
      260.0f
    );

  tft.drawFastHLine(
    30,
    236,
    260,
    TFT_DARKGREY
  );

  if (barW > 0) {
    tft.drawFastHLine(
      30,
      236,
      barW,
      rpmColor(d.rpm)
    );

    tft.drawFastHLine(
      30,
      237,
      barW,
      rpmColor(d.rpm)
    );
  }

  tft.setTextDatum(
    top_left
  );
}

// =========================================================
// SCREEN MANAGER
// =========================================================

void prepareCurrentScreen() {
  if (
    currentScreen ==
    SCREEN_SAMURAI
  ) {
    drawSamuraiStatic();
  }

  screenNeedsRedraw =
    false;
}

void updateCurrentScreen(
  const EngineData &d
) {
  if (
    screenNeedsRedraw
  ) {
    prepareCurrentScreen();
  }

  switch (
    currentScreen
  ) {
    case SCREEN_DASHBOARD:
      drawDashboard(d);
      break;

    case SCREEN_RPM:
      drawRPMScreen(d);
      break;

    case SCREEN_TELEMETRY:
      drawTelemetry(d);
      break;

    case SCREEN_BARS:
      drawBarsScreen(d);
      break;

    case SCREEN_SAMURAI:
      drawSamuraiScreen(d);
      break;

    default:
      break;
  }
}

// =========================================================
// SERIAL DEBUG
// =========================================================

void printEngineDataSerial(
  const EngineData &d
) {
  Serial.printf(
    "RPM: %u | "
    "TPS: %.1f%% | "
    "MAP: %u kPa | "
    "Lambda: %.2f | "
    "Duty: %u%% | "
    "IAT: %d C | "
    "CLT: %d C | "
    "GammaE: %u%% | "
    "Batt: %.1f V\n",
    d.rpm,
    d.tps,
    d.mapKpa,
    d.lambda,
    d.duty,
    d.iatC,
    d.cltC,
    d.gammaE,
    d.battV
  );
}

// =========================================================
// ECU TIMEOUT
// =========================================================

void drawECUTimeout() {
  tft.fillScreen(
    TFT_BLACK
  );

  tft.setTextDatum(
    middle_center
  );

  tft.setTextColor(
    TFT_RED,
    TFT_BLACK
  );

  tft.setTextSize(2);

  tft.drawString(
    "ECU TIMEOUT",
    160,
    110
  );

  tft.setTextSize(1);

  tft.setTextColor(
    TFT_DARKGREY,
    TFT_BLACK
  );

  tft.drawString(
    "CHECK SERIAL CONNECTION",
    160,
    135
  );

  tft.setTextDatum(
    top_left
  );
}

// =========================================================
// SETUP
// =========================================================

void setup() {
  Serial.begin(
    115200
  );

  randomSeed(
    esp_random()
  );

  pinMode(
    BUTTON_PIN,
    INPUT_PULLUP
  );

  if (!DEV_MODE) {
    ecu.begin(
      115200,
      SERIAL_8N1,
      ECU_RX,
      ECU_TX
    );
  }

  tft.init();
  tft.setRotation(1);
  tft.setTextWrap(false);

  // -------------------------------------------------------
  // UI BUFFER FOR NON-GIF SCREENS
  // -------------------------------------------------------

  uiSprite.setColorDepth(
    8
  );

  if (
    uiSprite.createSprite(
      320,
      240
    )
  ) {
    uiSpriteReady =
      true;

    uiSprite.setTextWrap(
      false
    );

    Serial.printf(
      "UI sprite allocated: %u bytes\n",
      (unsigned)
        uiSprite.bufferLength()
    );
  } else {
    Serial.println(
      "ERROR: UI sprite allocation failed"
    );
  }

  // -------------------------------------------------------
  // GIF / LITTLEFS - SAME AS OLD WORKING VERSION
  // -------------------------------------------------------

  gif.begin(
    LITTLE_ENDIAN_PIXELS
  );

  littleFSReady =
    LittleFS.begin(false);

  if (!littleFSReady) {
    Serial.println(
      "LittleFS mount FAILED"
    );

    Serial.println(
      "Upload the data folder to LittleFS."
    );
  } else {
    Serial.println(
      "LittleFS mounted"
    );

    if (
      LittleFS.exists(
        SAMURAI_GIF_PATH
      )
    ) {
      File f =
        LittleFS.open(
          SAMURAI_GIF_PATH,
          "r"
        );

      Serial.printf(
        "Found %s (%u bytes)\n",
        SAMURAI_GIF_PATH,
        (unsigned)f.size()
      );

      f.close();
    } else {
      Serial.printf(
        "Missing %s\n",
        SAMURAI_GIF_PATH
      );
    }
  }

  currentScreen =
    SCREEN_DASHBOARD;

  screenNeedsRedraw =
    true;
}

// =========================================================
// LOOP
// =========================================================

void loop() {
  handleButton();

  bool valid;

  if (DEV_MODE) {
    generateDevelopmentData(
      currentData
    );

    valid = true;
  } else {
    valid =
      requestSpeeduinoData(
        currentData
      );
  }

  if (valid) {
    updateCurrentScreen(
      currentData
    );

    printEngineDataSerial(
      currentData
    );
  } else {
    Serial.println(
      "ECU TIMEOUT"
    );

    stopSamuraiGif();

    drawECUTimeout();

    screenNeedsRedraw =
      true;
  }

  delay(20);
}
