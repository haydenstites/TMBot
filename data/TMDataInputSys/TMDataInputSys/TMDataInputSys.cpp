// TODO: Possibly clean up WASD vars to arrays

#define WINVER 0x0500
#include "pch.h"
#include "framework.h"
#include "TMDataInputSys.h"

// W = 0x11 , A = 0x1e, S = 0x1f, D = 0x20
UINT w_scan = 0x11; // 17
UINT a_scan = 0x1e; // 30
UINT s_scan = 0x1f; // 31
UINT d_scan = 0x20; // 32

// Boilerplate input setup
INPUT KeyInput(UINT scanCode) {
    INPUT input;

    input.type = INPUT_KEYBOARD;
    input.ki.wScan = scanCode; // Scan code e.g. 0x11 (https://www.win.tue.nl/~aeb/linux/kbd/scancodes-1.html)
    input.ki.time = 0;
    input.ki.dwExtraInfo = 0;
    input.ki.wVk = 0;

    return input;
}

void KeyDown(UINT scanCode) {
    INPUT input = KeyInput(scanCode);

    input.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &input, sizeof(INPUT));
}

void KeyUp(UINT scanCode) {
    INPUT input = KeyInput(scanCode);

    input.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &input, sizeof(INPUT));
}

void HandleKey(bool key_down, UINT scanCode) {
    if (key_down) {
        KeyDown(scanCode);
    } else {
        KeyUp(scanCode);
    }
}

// Input keys that should be pressed, reset_in scan code provided on reset
TMDATAINPUTSYS_API void SpoofKeys(int inputY, int inputX, bool enter, bool reset, bool escape, UINT enter_scan, UINT reset_scan, UINT escape_scan)
{
    HandleKey(enter, enter_scan);

    HandleKey(escape, escape_scan);
    HandleKey(reset, reset_scan);

    bool w = (inputY > 0);
    bool a = (inputX < 0);
    bool s = (inputY < 0);
    bool d = (inputX > 0);

    HandleKey(w, w_scan);
    HandleKey(a, a_scan);
    HandleKey(s, s_scan);
    HandleKey(d, d_scan);
}
