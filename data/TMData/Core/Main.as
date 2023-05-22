string path_dll = IO::FromStorageFolder("TMDataInputSys.dll");
Import::Library@ inputSys = Import::GetLibrary(path_dll);
Import::Function@ SpoofKeys = inputSys.GetFunction("SpoofKeys"); // in : (bool w, bool a, bool s, bool d)

bool switchingMap = false;
bool isPaused = false;
int map_epochs = 0;

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
	if (switchingMap) {
		SwitchMap();
	}
}

void RenderMenu()
{
  	if (UI::MenuItem("Load random train map")) {
		switchingMap = true;

		CTrackMania@ app = cast<CTrackMania>(GetApp());
		app.BackToMainMenu();
  	}
}
