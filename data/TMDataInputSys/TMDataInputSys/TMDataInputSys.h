#define TMDATAINPUTSYS_API extern "C" __declspec(dllexport)

TMDATAINPUTSYS_API void SpoofKeys(int inputY, int inputX, bool enter, bool reset, bool escape, UINT enter_scan, UINT reset_scan, UINT escape_scan);
