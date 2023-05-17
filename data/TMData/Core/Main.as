// string path_dll = IO::FromDataFolder("Plugins\\TMData\\TMDataInputSys\\x64\\Debug\\TMDataInputSys.dll");
string path_dll = IO::FromDataFolder("PluginStorage\\TMData\\TMDataInputSys.dll");
Import::Library@ inputSys = Import::GetLibrary(path_dll);
Import::Function@ SpoofKeys = inputSys.GetFunction("SpoofKeys"); // in : (bool w, bool a, bool s, bool d)

void Main() {
	IO::File f(IO::FromStorageFolder(Setting_IO_AltName), IO::FileMode::Write);
	for (int i = 0; i < 2; i++) {
    	f.WriteLine(tostring(0));
  	}
	f.Close();
}

void Update(float dt) {
	CSceneVehicleVisState@ state = VehicleState::ViewingPlayerState();
	if (!(state is null)) {
		Input();
		Output(state);
	}
} 
