[Setting category="General" name="Output"]
bool Setting_General_Output = true;

[Setting category="General" name="Input"]
bool Setting_General_Input = true;


[Setting category="IO" name="Output File Name"]
string Setting_IO_OutName = "state.txt";

[Setting category="IO" name="Input File Name"]
string Setting_IO_InName = "in.txt";

[Setting category="IO" name="Alt File Name"]
string Setting_IO_AltName = "alt.txt";


[Setting category="Output" name="Observations" description="Observations necessary for TMBot"]
bool Setting_Output_Observations = true;

[Setting category="Output" name="Extra Observations" description="Observations useful for reward shaping, used by default"]
bool Setting_Output_RewObservations = true;

[Setting category="Output" name="Checkpoints" description="From Checkpoint Counter"]
bool Setting_Output_Checkpoints = true;

[Setting category="Output" name="Bonk" description="From Bonk"]
bool Setting_Output_Bonk = true;


enum ControlType
{
	Keyboard,
	Gamepad,
}

[Setting category="Input" name="Controller Type"]
ControlType Setting_Input_Control = ControlType::Keyboard;

[Setting category="Input" name="Deadzone" description="Note: Applies to both Keyboard and Gamepad control types."]
bool Setting_Input_Deadzone = true;

[Setting category="Input" name="Deadzone Magnitude" description="Note: Applies to both Keyboard and Gamepad control types." drag min=0 max=1]
float Setting_Input_DeadzoneMag = 0.1f;

[Setting category="Input" name="Reset" description="Control when game resets."]
bool Setting_Input_Reset = true;

[Setting category="Input" name="Reset Scancode" description="Physical keyboard scan code."]
uint Setting_Input_ResetScan = 47;

[Setting category="Input" name="Pause" description="Control when game pauses."]
bool Setting_Input_Escape = true;

[Setting category="Input" name="Pause Scancode" description="Physical keyboard scan code."]
uint Setting_Input_EscapeScan = 01;
