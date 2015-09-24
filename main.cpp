#include <iostream>
#define WINVER 0x0500
#include <windows.h>

using namespace std;

//key reference
//https://msdn.microsoft.com/en-us/library/windows/desktop/dd375731(v=vs.85).aspx

INPUT ip;
void init(){
    ip.type = INPUT_KEYBOARD;
    ip.ki.wScan = 0; // hardware scan code for key
    ip.ki.time = 0;
    ip.ki.dwExtraInfo = 0;
}
void press(int code){
    // Press the "A" key
    ip.ki.wVk = code; // virtual-key code for the "a" key
    ip.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ip, sizeof(INPUT));

    // Release the "A" key
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));
}

int main()
{
    cout<<"Windows Virtual Key API Tester"<<endl;
    cout<<"Press a) for Volume UP"<<endl;
    cout<<"Press s) for Volume DOWN"<<endl;
    cout<<"Press d) for Volume MUTE/UNMUTE"<<endl;
    cout<<"Press f) for PLAY/PAUSE"<<endl;
    cout<<"Press g) for STOP"<<endl;
    cout<<"Press h) for PREVIOUS"<<endl;
    cout<<"Press j) for NEXT"<<endl;

    init();

    char ch;
    while(1){
        cin>>ch;
        switch(ch){
            case 'a': press(0xAF); break;
            case 's': press(0xAE); break;
            case 'd': press(0xAD); break;
            case 'f': press(0xB3); break;
            case 'g': press(0xA9); break;
            case 'h': press(0xB1); break;
            case 'j': press(0xB0); break;
            default : cout<<"---"<<endl; break;
        }

    }
    return 0;
}
