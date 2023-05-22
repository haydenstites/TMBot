void Output(CSceneVehicleVisState@ vis) {
	if (Setting_General_Output) {
		CTrackMania@ app = cast<CTrackMania>(GetApp());
		if (tostring(app.CurrentPlayground.GameTerminals[0].UISequence_Current) == "Playing") {
			IO::File f(IO::FromStorageFolder(Setting_IO_OutName), IO::FileMode::Write);

			if (Setting_Output_Observations) {
				f.WriteLine(tostring(vis.FrontSpeed * 3.6f));
				f.WriteLine(tostring(VehicleState::GetSideSpeed(vis) * 3.6f));
				f.WriteLine(tostring(vis.CurGear));
				f.WriteLine(tostring(vis.IsWheelsBurning));
				f.WriteLine(tostring(vis.FLGroundContactMaterial));
				f.WriteLine(tostring(vis.IsGroundContact));
			}
			if (Setting_Output_RewObservations) {
				f.WriteLine(tostring(vis.IsTopContact));
				f.WriteLine(tostring(app.CurrentPlayground.GameTerminals[0].UISequence_Current));
				f.WriteLine(tostring(app.RootMap.TMObjective_AuthorTime));
				f.WriteLine(tostring(Time::Now));
			}
			if (Setting_Output_Checkpoints) { // And plugin installed
				f.WriteLine(tostring(CP::curCP));
				f.WriteLine(tostring(CP::maxCP));
			}
			if (Setting_Output_Bonk) { // And plugin installed
				f.WriteLine(tostring(Bonk::lastBonkTime()));
			}

			f.Close();
		}
	}
}

void Input() {
	string path_in = IO::FromStorageFolder(Setting_IO_InName);
	if (Setting_General_Input && IO::FileExists(path_in)) {
		CTrackMania@ app = cast<CTrackMania>(GetApp());

		IO::File f(path_in, IO::FileMode::Read);

		float inputY = Deadzone(Text::ParseFloat(f.ReadLine()), Setting_Input_DeadzoneMag);
		float inputX = Deadzone(Text::ParseFloat(f.ReadLine()), Setting_Input_DeadzoneMag);

		f.Close();

		int reset = 0;
		int escape = 0;
		string path_alt = IO::FromStorageFolder(Setting_IO_AltName);
		if ((Setting_Input_Reset || Setting_Input_Escape) && IO::FileExists(path_alt)) {
			IO::File fr_read(path_alt, IO::FileMode::Read); // fr fr
			reset = Text::ParseInt(fr_read.ReadLine());
			escape = Text::ParseInt(fr_read.ReadLine());
			fr_read.Close();
			if (reset == 1 || escape == 1) {
				IO::File fr_write(path_alt, IO::FileMode::Write);
				fr_write.WriteLine(tostring(0)); // Reset
				fr_write.WriteLine(tostring(0)); // Escape
				fr_write.Close();
			}
		}

		bool enter = false;
		if (Setting_Input_Enter) {
			string cur_seq = tostring(app.CurrentPlayground.GameTerminals[0].UISequence_Current);
			enter = (cur_seq != "Playing");
		}

		if (Setting_Input_Control == ControlType::Keyboard) {
			SpoofKeys.Call(int(inputY), int(inputX), enter, reset, escape, Setting_Input_EnterScan, Setting_Input_ResetScan, Setting_Input_EscapeScan);
		}

		if (escape == 1) {
			if (isPaused) {
				isPaused = false;

				map_epochs += 1;
				if (map_epochs >= Setting_Input_MapEpochs) {
					map_epochs = 0;
					
					switchingMap = true;
					app.BackToMainMenu();
				}
			}
			else {
				isPaused = true;
			}
		}
	}
}

void SwitchMap() {
	CTrackMania@ app = cast<CTrackMania>(GetApp());

	if (app.ManiaTitleControlScriptAPI.IsReady) {
		IO::File f(IO::FromStorageFolder(Setting_IO_MapName), IO::FileMode::Read);
		int len = Text::ParseInt(f.ReadLine());

		if (len == 0) {
			switchingMap = false;
		}
		else {
			int index = Math::Rand(0, len);
			string mapID = "NA";

			for (int i = 0; i <= index; i++) {
				mapID = f.ReadLine();
			}

			print(mapID);

			app.ManiaTitleControlScriptAPI.PlayMap(mapID, "", "");

			switchingMap = false;
		}
		f.Close();
    }
}

float Deadzone(float num, float mag) {
	if (Setting_Input_Deadzone && (Math::Abs(num) <= mag)) {
		return 0;
	}
	return num;
}
