The meta.csv file records the full annotation information of the audio files.

Specifically, the file contains 9 columns:

Column 1: Unique File ID.

Column 2: Speech Type ID (1 = source, 2 = genuine, 3 = replayed).

Column 3: Speaker ID, Unique Speaker Identifier (range: 0-50; 0 = tts speaker)

Column 4: Environment ID (1 = outdoor, 2 = indoor environment 1, 3 = indoor environment 2, 4 = vehicle environment), see Figure 2 in the paper for details.

Column 5：Position ID 
			Env1: 1 = 0.5m, 2 = 1.5m, -1 = unknown.
			Env2: In the form of 'AB'; First digit (A): position of the mic arrays (range: 1-3); Second digit (B): position of the speaker (range: 1-6).
			Env3: -1, speaker position is fixed.
			Env4: Genine Recordings: In the form of 'AB'; First digit (A): 1 = quite 2 = moving; Second digit (B): position of the speaker (range: 1-6)
				  Replayed Recording: 0 = vehicle built-in speaker position, 1-6 = position shown in the Figure 4 of the paper.

Column 6：Source Recorder ID ( -1 = N/A, 1 = DR_05, 2 = iPod, 3 = TTS)

Column 7: Playback Device ID ( -1 = N/A, 1 = Audio Technica ATH-AD700X headphone, 2 = iPod Touch, 3 = Sony SRS_X11, 4 = Sony SRS_X5, 5 = Vehicle built-in speaker)

Column 8: Recording Device ID ( -1 = N/A, 1 = AIY, 2 = Respeaker 4_Linear, 3 = Respeaker Core V2.0, 4 = Amlogic A113X1)

Column 9: Recording Length (Seconds)