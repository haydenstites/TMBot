#define TMDATAINPUTSYS_API extern "C" __declspec(dllexport)

TMDATAINPUTSYS_API void SpoofKeys(bool w, bool a, bool s, bool d, bool enter, bool reset, bool escape, UINT reset_scan, UINT escape_scan);
