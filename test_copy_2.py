import sys
import pyaudio
import wave
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pydub import AudioSegment
# import whisper
import pyautogui
import keyboard
import time
import torch
from RealtimeSTT import AudioToTextRecorder
import RealtimeSTT

currentmic = 0

class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition Tool")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Set application icon and styling
        self.root.configure(bg="#f0f0f0")
        
        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.WAVE_OUTPUT_FILENAME = "recording.wav"
        self.MP3_OUTPUT_FILENAME = "recording.mp3"
        self.KEY_TO_HOLD = "r"  # Key to hold for recording
        
        # Real-time STT settings
        self.realtime_recorder = None
        self.is_realtime_mode = False
        
        # Initialize PyAudio and Whisper model
        self.audio = pyaudio.PyAudio()
        try:
            # Determine the path for Whisper model
            if getattr(sys, 'frozen', False):
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                application_path = sys._MEIPASS
                model_path = os.path.join(application_path, 'whisper_model')
            else:
                # For development environment
                application_path = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(application_path, 'whisper_model')
            
            # Ensure model directory exists
            os.makedirs(model_path, exist_ok=True)
            
            # Force CPU usage regardless of GPU availability
            device = "cuda"
            
            print(f"Using device: {device}")
            # Choose a Whisper model - options: tiny, base, small, medium, large
            # self.whisper_model = whisper.load_model("base", download_root=model_path, device=device)
            
            # Save device info for later
            self.device_info = f"Using {device.upper()} for inference with fp32 precision"
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Whisper model: {str(e)}")
            self.whisper_model = None
            self.device_info = "Failed to initialize model"
        
        # Variables for recording status
        self.is_recording = False
        self.is_listening_for_key = False
        self.recording_thread = None
        self.key_listener_thread = None
        self.frames = []
        self.stream = None
        self.waiting_for_position = False
        
        # UI Variables
        self.mic_devices = []
        self.selected_mic_var = tk.StringVar()
        self.position1 = None
        self.position2 = None
        self.status_var = tk.StringVar(value=self.device_info)  # Set initial status to device info
        self.transcription_var = tk.StringVar(value="")
        self.key_var = tk.StringVar(value=self.KEY_TO_HOLD.upper())
        
        # Auto-click checkbox variable
        self.auto_click_var = tk.BooleanVar(value=True)
        
        # Create UI
        self.create_widgets()
        
        # Load available microphones
        self.load_microphones()
        
        # Start listening for the R key
        self.start_key_listener()

    def create_widgets(self):
        """Create all UI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Speech Recognition Tool", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Recording mode frame
        mode_frame = ttk.LabelFrame(main_frame, text="Recording Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.mode_var = tk.BooleanVar(value=False)
        mode_label = ttk.Label(mode_frame, text="Real-time Mode:")
        mode_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        mode_checkbox = ttk.Checkbutton(mode_frame, text="Enable real-time transcription", 
                                      variable=self.mode_var, 
                                      command=self.toggle_recording_mode)
        mode_checkbox.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Microphone selection frame
        mic_frame = ttk.LabelFrame(main_frame, text="Microphone Selection")
        mic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        mic_label = ttk.Label(mic_frame, text="Select Microphone:")
        mic_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.mic_combobox = ttk.Combobox(mic_frame, textvariable=self.selected_mic_var, state="readonly", width=40)
        self.mic_combobox.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        self.mic_combobox.bind("<<ComboboxSelected>>", self.on_mic_selected)
        
        refresh_button = ttk.Button(mic_frame, text="Refresh", command=self.load_microphones)
        refresh_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Click positions frame
        pos_frame = ttk.LabelFrame(main_frame, text="Click Positions")
        pos_frame.pack(fill=tk.X, padx=10, pady=5)
        
        pos1_button = ttk.Button(pos_frame, text="Set Position 1", command=lambda: self.set_position(1))
        pos1_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.pos1_label = ttk.Label(pos_frame, text="Not set")
        self.pos1_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        pos2_button = ttk.Button(pos_frame, text="Set Position 2", command=lambda: self.set_position(2))
        pos2_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.pos2_label = ttk.Label(pos_frame, text="Not set")
        self.pos2_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Recording key frame
        key_frame = ttk.LabelFrame(main_frame, text="Recording Key")
        key_frame.pack(fill=tk.X, padx=10, pady=5)
        
        key_label = ttk.Label(key_frame, text="Hold this key to record:")
        key_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        key_entry = ttk.Entry(key_frame, textvariable=self.key_var, width=5)
        key_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        key_button = ttk.Button(key_frame, text="Apply", command=self.change_recording_key)
        key_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        instruction_label = ttk.Label(key_frame, text="Press key while focused outside this window")
        instruction_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.pack(padx=5, pady=5, anchor=tk.W)
        
        # Transcription frame
        transcription_frame = ttk.LabelFrame(main_frame, text="Transcription")
        transcription_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.transcription_text = tk.Text(transcription_frame, wrap=tk.WORD, height=8)
        self.transcription_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Settings frame
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(settings_frame, text="Auto click after transcription:").pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(settings_frame, variable=self.auto_click_var).pack(side=tk.LEFT)
        
        # Bottom status bar
        self.status_bar = ttk.Label(self.root, text=f"Ready - Hold '{self.KEY_TO_HOLD.upper()}' to record", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_microphones(self):
        """Load all available microphones into the combobox"""
        self.mic_devices = []
        self.mic_combobox["values"] = []
        
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        device_names = []
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                self.mic_devices.append((i, device_info.get('name')))
                device_names.append(device_info.get('name'))
        
        if device_names:
            self.mic_combobox["values"] = device_names
            self.mic_combobox.current(0)
            self.on_mic_selected(None)
            self.status_var.set(f"Found {len(device_names)} microphones")
        else:
            self.status_var.set("No microphones found")

    def on_mic_selected(self, event):
        """Handle microphone selection"""
        selected_name = self.selected_mic_var.get()
        for idx, name in self.mic_devices:
            if name == selected_name:
                self.selected_mic = idx
                global currentmic
                currentmic = idx
                self.status_var.set(f"Selected microphone: {name}")
                
                # If real-time mode is active, disable it
                if self.is_realtime_mode:
                    self.mode_var.set(False)
                    self.toggle_recording_mode()
                return

    def set_position(self, pos_num):
        """Set cursor position for text placement using Enter key"""
        if self.waiting_for_position:
            return  # Prevent multiple position setting operations
            
        self.waiting_for_position = True
        
        # Create a small popup window with instructions
        popup = tk.Toplevel(self.root)
        popup.title("Set Position")
        popup.geometry("400x150")
        popup.attributes('-topmost', True)
        
        instruction_text = f"""
        1. Move your cursor to the desired position {pos_num}
        2. Press Enter to confirm
        3. Keep this window open during the process
        """
        
        ttk.Label(popup, text=instruction_text, font=("Arial", 11)).pack(pady=20)
        ttk.Label(popup, text="Waiting for Enter key...", foreground="blue").pack()
        
        # Update status
        self.status_var.set(f"Move cursor to position {pos_num} and press Enter...")
        
        # Create a listener thread that waits for Enter key
        def position_listener():
            try:
                # Wait for Enter key
                keyboard.wait('enter', suppress=False)
                
                # Get the current position
                pos = pyautogui.position()
                
                # Update the UI from the main thread
                self.root.after(0, lambda: self._update_position(pos_num, pos, popup))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error setting position: {str(e)}"))
                self.waiting_for_position = False
                popup.destroy()
        
        # Start the listener thread
        listener_thread = threading.Thread(target=position_listener)
        listener_thread.daemon = True
        listener_thread.start()
        
        # Add a cancel button
        ttk.Button(popup, text="Cancel", 
                  command=lambda: self._cancel_position_setting(popup)).pack(pady=10)
        
        # When the popup is closed, end the waiting state
        popup.protocol("WM_DELETE_WINDOW", lambda: self._cancel_position_setting(popup))

    def _update_position(self, pos_num, pos, popup):
        """Update the position label and store the position"""
        if pos_num == 1:
            self.position1 = pos
            self.pos1_label.configure(text=f"X: {pos.x}, Y: {pos.y}")
        else:
            self.position2 = pos
            self.pos2_label.configure(text=f"X: {pos.x}, Y: {pos.y}")
        
        self.status_var.set(f"Position {pos_num} set to ({pos.x}, {pos.y})")
        self.waiting_for_position = False
        popup.destroy()

    def _cancel_position_setting(self, popup):
        """Cancel the position setting operation"""
        self.waiting_for_position = False
        self.status_var.set("Position setting cancelled")
        popup.destroy()

    def start_key_listener(self):
        """Start listening for the recording key press"""
        if self.is_listening_for_key:
            return
            
        self.is_listening_for_key = True
        
        # Start key listener thread
        self.key_listener_thread = threading.Thread(target=self._key_listener)
        self.key_listener_thread.daemon = True
        self.key_listener_thread.start()
        
        self.status_var.set(f"Listening for '{self.KEY_TO_HOLD.upper()}' key press")

    def stop_key_listener(self):
        """Stop listening for the recording key press"""
        self.is_listening_for_key = False
        
        # This doesn't actually stop the thread but marks it as no longer needed
        # The thread will exit on its next iteration
        self.status_var.set("Key listener stopped")

    def _key_listener(self):
        """Listen for key presses and start/stop recording accordingly"""
        while self.is_listening_for_key:
            try:
                # Wait for the key to be pressed
                keyboard.wait(self.KEY_TO_HOLD, suppress=False)
                
                # Start recording if key is pressed
                if keyboard.is_pressed(self.KEY_TO_HOLD) and not self.is_recording:
                    self.root.after(0, self.start_recording)
                    
                    # Wait for key to be released
                    while keyboard.is_pressed(self.KEY_TO_HOLD) and self.is_listening_for_key:
                        pass
                    
                    # Stop recording when key is released
                    if self.is_recording:
                        self.root.after(0, self.stop_recording)
                
                # Small delay to prevent CPU usage
                import time
                time.sleep(0.05)
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Key listener error: {str(e)}"))
                break
                
        # Exit message when thread terminates
        print("Key listener thread terminated")

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return
            
        if not hasattr(self, 'selected_mic'):
            self.status_var.set("Please select a microphone first")
            return
            
        self.is_recording = True
        self.status_var.set(f"Recording... (holding '{self.KEY_TO_HOLD.upper()}' key)")
        self.status_bar.config(text="Recording...", background="red")
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def _record(self):
        """Record audio in a background thread"""
        self.frames = []
        try:
            self.stream = self.audio.open(
                format=self.FORMAT, 
                channels=self.CHANNELS,
                rate=self.RATE, 
                input=True, 
                input_device_index=self.selected_mic,
                frames_per_buffer=self.CHUNK
            )
            
            while self.is_recording and keyboard.is_pressed(self.KEY_TO_HOLD):
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                
            # If we're still recording but the key was released, stop recording
            if self.is_recording:
                self.root.after(0, self.stop_recording)
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Recording error: {str(e)}"))
            self.is_recording = False

    def stop_recording(self):
        """Stop recording and process the audio"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.status_var.set("Processing audio...")
        self.status_bar.config(text="Processing...", background="#f0f0f0")
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Process the recording in a separate thread
        processing_thread = threading.Thread(target=self._process_recording)
        processing_thread.daemon = True
        processing_thread.start()

    def _process_recording(self):
        """Save and process the recording"""
        try:
            # Don't process if no frames were recorded
            if not self.frames:
                self.root.after(0, lambda: self.status_var.set("No audio recorded"))
                self.root.after(0, lambda: self.status_bar.config(text=f"Ready - Hold '{self.KEY_TO_HOLD.upper()}' to record", background="#f0f0f0"))
                return
                
            # Save as WAV file
            with wave.open(self.WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.frames))
            
            # Convert to MP3 if needed
            try:
                sound = AudioSegment.from_wav(self.WAVE_OUTPUT_FILENAME)
                sound.export(self.MP3_OUTPUT_FILENAME, format="mp3")
                self.root.after(0, lambda: self.status_var.set("Audio saved"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"MP3 conversion error: {str(e)}"))
            
            # Transcribe audio
            self.root.after(0, lambda: self.status_var.set("Transcribing..."))
            self._transcribe_audio()
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Processing error: {str(e)}"))
            self.root.after(0, lambda: self.status_bar.config(text=f"Ready - Hold '{self.KEY_TO_HOLD.upper()}' to record", background="#f0f0f0"))

    def _transcribe_audio(self):
        """Convert speech to text using Whisper and handle the result"""
        if not os.path.exists(self.WAVE_OUTPUT_FILENAME):
            self.root.after(0, lambda: self.status_var.set("No recording found"))
            self.root.after(0, lambda: self.status_bar.config(text=f"Ready - Hold '{self.KEY_TO_HOLD.upper()}' to record", background="#f0f0f0"))
            return
            
        try:
            # Check if Whisper model is loaded
            if not self.whisper_model:
                raise Exception("Whisper model not loaded")
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(self.WAVE_OUTPUT_FILENAME)
            text = result['text'].strip()
            
            # Update UI with transcription
            self.root.after(0, lambda: self._update_transcription(text))
            
            # Perform click and type operations if enabled
            if self.auto_click_var.get() and self.position1 and self.position2:
                pyautogui.moveTo(self.position1)
                pyautogui.doubleClick()
                pyautogui.typewrite(text)
                pyautogui.moveTo(self.position2)
                pyautogui.click()
                
                self.root.after(0, lambda: self.status_var.set("Transcribed and inserted text"))
            else:
                self.root.after(0, lambda: self.status_var.set("Transcription complete"))
                    
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Transcription error: {str(e)}"))
            
        finally:
            self.root.after(0, lambda: self.status_bar.config(text=f"Ready - Hold '{self.KEY_TO_HOLD.upper()}' to record", background="#f0f0f0"))

    def _update_transcription(self, text):
        """Update the transcription text box"""
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, text)

    def change_recording_key(self):
        """Change the key used for recording"""
        new_key = self.key_var.get().lower()
        if new_key and len(new_key) == 1:
            self.KEY_TO_HOLD = new_key
            self.status_var.set(f"Recording key changed to '{new_key.upper()}'")
            self.status_bar.config(text=f"Ready - Hold '{new_key.upper()}' to record")
            
            # Restart key listener with new key
            self.stop_key_listener()
            self.start_key_listener()
        else:
            self.status_var.set("Please enter a single character for the recording key")
            self.key_var.set(self.KEY_TO_HOLD.upper())

    def toggle_recording_mode(self):
        """Toggle between real-time and key-hold recording modes"""
        self.is_realtime_mode = self.mode_var.get()
        if self.is_realtime_mode:
            self.stop_key_listener()
            self.start_realtime_recording()
        else:
            self.stop_realtime_recording()
            self.start_key_listener()

    def start_realtime_recording(self):
        """Start real-time speech-to-text recording"""
        if not hasattr(self, 'selected_mic'):
            self.status_var.set("Please select a microphone first")
            self.mode_var.set(False)
            return
            
        try:
            # Stop any existing recorder first
            if self.realtime_recorder:
                self.stop_realtime_recording()
            
            # Create new recorder
            self.realtime_recorder = AudioToTextRecorder(input_device_index=currentmic)
            self.status_var.set("Real-time recording started - Speak now")
            self.status_bar.config(text="Real-time recording active", background="green")
            
            # Start processing in a separate thread
            threading.Thread(target=self._process_realtime_text, daemon=True).start()
        except Exception as e:
            self.status_var.set(f"Failed to start real-time recording: {str(e)}")
            self.mode_var.set(False)
            self.stop_realtime_recording()

    def stop_realtime_recording(self):
        """Stop real-time speech-to-text recording"""
        if self.realtime_recorder:
            try:
                # Stop the recorder
                self.realtime_recorder.shutdown()
                # Clear the recorder reference
                self.realtime_recorder = None
                self.status_var.set("Real-time recording stopped")
                self.status_bar.config(text=f"Ready - Hold '{self.KEY_TO_HOLD.upper()}' to record", background="#f0f0f0")
            except Exception as e:
                print(f"Error stopping recorder: {str(e)}")
                self.realtime_recorder = None
                self.status_var.set("Error stopping recording")
                self.status_bar.config(text=f"Ready - Hold '{self.KEY_TO_HOLD.upper()}' to record", background="#f0f0f0")

    def _process_realtime_text(self):
        """Process real-time transcribed text in a continuous loop"""
        def on_text(text):
            if text.strip():
                self.root.after(0, lambda: self._update_transcription(text))
                if self.auto_click_var.get() and self.position1 and self.position2:
                    self.root.after(0, lambda: self._perform_auto_click(text))
        
        try:
            while self.is_realtime_mode and self.realtime_recorder:
                try:
                    self.realtime_recorder.text(on_text)
                except Exception as e:
                    print(f"Error in recorder.text(): {str(e)}")
                    break
                # Small delay to prevent CPU overuse
                import time
                time.sleep(0.1)
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Real-time processing error: {str(e)}"))
        finally:
            # Ensure we stop the recording when the loop ends
            self.root.after(0, lambda: self.stop_realtime_recording())

    def _perform_auto_click(self, text):
        """Perform auto-click operations for the transcribed text"""
        pyautogui.moveTo(self.position1)
        pyautogui.doubleClick()
        pyautogui.typewrite(text)
        pyautogui.moveTo(self.position2)
        pyautogui.click()

    def on_closing(self):
        """Handle the window closing event"""
        self.stop_key_listener()
        self.stop_realtime_recording()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()