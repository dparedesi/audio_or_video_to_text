import tkinter as tk
from tkinter import ttk
import pyaudio
import wave
import threading
from faster_whisper import WhisperModel
from pathlib import Path
from datetime import datetime
import queue
import time
import tempfile
import os
import numpy as np
import collections
import webrtcvad
import warnings

# Suppress numpy warnings from faster-whisper
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')

class TranscriptionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Live Voice Transcriber")
        self.window.geometry("700x650")
        self.window.configure(bg="#f0f0f0")  # Light gray background for visibility
        self.recording = False
        self.full_transcription = []
        self.audio_queue = queue.Queue()  # Queue for audio chunks
        self.chunk_counter = 0
        self.current_audio_level = 0
        self.is_processing = False
        self.segments_recorded = 0
        self.segments_transcribed = 0
        
        # Full audio recording
        self.full_audio_frames = []  # Store ALL audio for final transcription
        self.full_audio_file = None  # Temp file path for complete recording
        self.final_complete_transcription = ""  # Store the final high-quality transcription
        
        # Audio buffer for VAD
        self.audio_buffer = collections.deque(maxlen=int(16000 * 30))  # 30 seconds max buffer
        self.silence_duration = 0
        self.speech_started = False
        
        print("Loading Faster-Whisper model... (this takes a moment)")
        # Use CTranslate2 for faster inference
        # compute_type: "int8" for CPU, "float16" for GPU
        # Note: faster-whisper "small" model ≈ original "medium" in quality but much faster
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Model loaded!")
        
        print("Initializing VAD (Voice Activity Detection)...")
        # WebRTC VAD: lightweight, fast, no downloads needed
        # Mode 3 = most aggressive filtering (best for detecting pauses)
        self.vad = webrtcvad.Vad(mode=3)
        print("VAD initialized!")
        
        self.record_btn = tk.Button(
            self.window, text="🎤 Start Recording", 
            command=self.toggle_recording,
            width=20, height=2, font=("Arial", 14)
        )
        self.record_btn.pack(pady=20)
        
        # Audio Level Meter Frame
        meter_frame = tk.Frame(self.window)
        meter_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(meter_frame, text="Audio Level:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.audio_level_canvas = tk.Canvas(meter_frame, width=400, height=20, bg="#1a1a1a", 
                                           highlightthickness=2, highlightbackground="#555555")
        self.audio_level_canvas.pack(side=tk.LEFT, padx=5)
        self.audio_level_bar = self.audio_level_canvas.create_rectangle(0, 0, 0, 20, fill="#00ff00", outline="")
        
        self.audio_level_label = tk.Label(meter_frame, text="--", font=("Arial", 9), width=8)
        self.audio_level_label.pack(side=tk.LEFT, padx=5)
        
        # Processing Status Frame
        status_frame = tk.Frame(self.window)
        status_frame.pack(pady=5, padx=20, fill=tk.X)
        
        self.processing_label = tk.Label(status_frame, text="⏸ Idle", font=("Arial", 10), fg="gray")
        self.processing_label.pack(side=tk.LEFT, padx=10)
        
        self.queue_label = tk.Label(status_frame, text="", font=("Arial", 10), fg="gray")
        self.queue_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(self.window, text="Transcription:", font=("Arial", 12)).pack()
        
        # Text area with scrollbar and visible colors
        text_frame = tk.Frame(self.window)
        text_frame.pack(padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_area = tk.Text(text_frame, width=80, height=25, wrap=tk.WORD,
                                bg="white", fg="black", insertbackground="black",
                                font=("Arial", 11), yscrollcommand=scrollbar.set)
        self.text_area.pack(side=tk.LEFT)
        scrollbar.config(command=self.text_area.yview)
        
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=5)
        
        self.clear_btn = tk.Button(
            btn_frame, text="Clear Text",
            command=self.clear_text,
            width=15
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(
            btn_frame, text="Save Transcription",
            command=self.save_transcription,
            width=20
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(self.window, text="Ready", font=("Arial", 10), fg="green")
        self.status_label.pack(pady=5)
        
    def _update_text_area(self, text):
        """Thread-safe method to update text area - MUST be called from main thread"""
        print(f"[GUI]: '{text[:100]}...'")  # Debug
        try:
            self.text_area.insert(tk.END, text + "\n")
            self.text_area.see(tk.END)
            # Only use update_idletasks to avoid blocking issues
            self.text_area.update_idletasks()
        except Exception as e:
            print(f"[GUI] Error updating text area: {e}")
    
    def update_audio_level(self, level):
        """Update the audio level meter"""
        # Normalize level to 0-100
        level = max(0, min(100, level))
        bar_width = int(400 * level / 100)
        
        # Color based on level: green (good), yellow (low), red (too loud)
        if level < 5:
            color = "#555555"  # Very low/silence
        elif level < 30:
            color = "#ff9900"  # Low - speak louder
        elif level < 80:
            color = "#00ff00"  # Good range
        else:
            color = "#ff0000"  # Too loud
        
        self.audio_level_canvas.coords(self.audio_level_bar, 0, 0, bar_width, 20)
        self.audio_level_canvas.itemconfig(self.audio_level_bar, fill=color)
        
        # Update label
        if level < 5:
            status = "Silence"
        elif level < 30:
            status = "Low"
        elif level < 80:
            status = "Good"
        else:
            status = "High"
        self.audio_level_label.config(text=status)
    
    def update_status_display(self):
        """Update the real-time status display"""
        while self.recording or self.is_processing:
            try:
                # Update audio level meter
                self.window.after(0, lambda: self.update_audio_level(self.current_audio_level))
                
                # Update processing status
                if self.is_processing:
                    self.window.after(0, lambda: self.processing_label.config(text="🔄 Processing...", fg="#ff9900"))
                else:
                    if self.recording:
                        self.window.after(0, lambda: self.processing_label.config(text="✓ Listening", fg="#00cc00"))
                    else:
                        self.window.after(0, lambda: self.processing_label.config(text="⏸ Idle", fg="gray"))
                
                # Update queue status
                queue_size = self.audio_queue.qsize()
                if queue_size > 0:
                    queue_text = f"⏳ Pending: {queue_size} segment{'s' if queue_size > 1 else ''}"
                    self.window.after(0, lambda txt=queue_text: self.queue_label.config(text=txt, fg="#ff9900"))
                else:
                    if self.segments_recorded > 0:
                        total_text = f"✓ Processed: {self.segments_transcribed}/{self.segments_recorded}"
                        self.window.after(0, lambda txt=total_text: self.queue_label.config(text=txt, fg="#00cc00"))
                    else:
                        self.window.after(0, lambda: self.queue_label.config(text=""))
                
                time.sleep(0.1)  # Update 10 times per second
                
            except Exception as e:
                pass  # Silently ignore status update errors
                time.sleep(0.5)
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.full_transcription = []
        self.status_label.config(text="Text cleared", fg="blue")
    
    def save_transcription(self):
        # Save the final complete transcription (not the chunks)
        if not self.final_complete_transcription:
            self.status_label.config(text="No final transcription yet. Stop recording first.", fg="orange")
            return
        
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"outputs/final_transcription_{timestamp}.txt"
        Path("outputs").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(self.final_complete_transcription)
        
        self.status_label.config(text=f"Saved: {filename}", fg="green")
        print(f"Final transcription saved to {filename}")
    
    def toggle_recording(self):
        
        if not self.recording:
            self.recording = True
            self.chunk_counter = 0
            self.segments_recorded = 0
            self.segments_transcribed = 0
            self.full_audio_frames = []  # Reset full audio storage
            self.final_complete_transcription = ""  # Reset final transcription
            # Clear the queue
            while not self.audio_queue.empty():
                self.audio_queue.get()
            
            self.record_btn.config(text="⏹️ Stop Recording", bg="red", fg="white")
            self.status_label.config(text="Recording...", fg="red")
            
            # Add initial feedback message
            self.text_area.insert(tk.END, "🎤 Recording started... Speak now!\n\n")
            self.text_area.see(tk.END)
            
            # Start all worker threads
            threading.Thread(target=self.record_audio, daemon=True).start()
            threading.Thread(target=self.transcribe_audio, daemon=True).start()
            threading.Thread(target=self.update_status_display, daemon=True).start()
        else:
            self.recording = False
            self.record_btn.config(text="🎤 Start Recording", bg="SystemButtonFace", fg="black")
            self.status_label.config(text="Processing full audio...", fg="orange")
            self.processing_label.config(text="⏸ Idle", fg="gray")
            self.queue_label.config(text="")
            self.update_audio_level(0)
            
            # Process the full audio in a separate thread
            threading.Thread(target=self.process_full_audio, daemon=True).start()
    
    def record_audio(self):
        """Simple time-based recording - records in 5-second chunks"""
               
        audio = None
        try:
            audio = pyaudio.PyAudio()
        except Exception as e:
            return  # Silently fail
        
        try:
            # Simple time-based recording configuration
            chunk_duration = 5  # Record in 5-second chunks
            chunk_size = 1024
            sample_rate = 16000
            
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
            
            while self.recording:
                frames = []
                
                # Record for chunk_duration seconds
                frames_to_record = int(sample_rate / chunk_size * chunk_duration)
                
                for _ in range(frames_to_record):
                    if not self.recording:
                        break
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    self.full_audio_frames.append(data)  # Store for final processing
                    
                    # Update audio level display
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    level = np.abs(audio_data).mean() / 32768.0 * 100
                    self.current_audio_level = level
                
                if frames:
                    self.chunk_counter += 1
                    # Put chunk in queue for transcription (chunk_num, frames, sample_width)
                    self.audio_queue.put((self.chunk_counter, frames, audio.get_sample_size(pyaudio.paInt16)))
                    self.segments_recorded += 1
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            error_msg = f"\n\n[RECORDING] Error: {str(e)}\n"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.window.after(0, lambda: self.text_area.insert(tk.END, error_msg))
        finally:
            try:
                audio.terminate()
            except:
                pass
    
    def transcribe_audio(self):
        """Transcription thread - processes speech segments from queue using faster-whisper"""
        while self.recording or not self.audio_queue.empty():
            temp_file = None
            try:
                # Wait for audio segments with timeout
                chunk_num, frames, sample_width = self.audio_queue.get(timeout=1)
                
                self.is_processing = True
                
                # Create temporary file that will be auto-deleted
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                filename = temp_file.name
                temp_file.close()  # Close so we can write with wave module
                
                # Validate and normalize audio before saving
                audio_bytes = b''.join(frames)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Check if audio has actual content
                if audio_array.size == 0 or np.abs(audio_array).max() < 100:
                    self.audio_queue.task_done()
                    self.is_processing = False
                    continue
                
                # Simple energy check - skip completely silent chunks
                rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                if rms < 50:  # Very low threshold - only skip true silence
                    self.audio_queue.task_done()
                    self.is_processing = False
                    continue
                
                # Normalize audio to prevent numerical issues
                audio_max = np.abs(audio_array).max()
                if audio_max > 0:
                    normalized_audio = (audio_array.astype(np.float32) / audio_max * 32767 * 0.95).astype(np.int16)
                else:
                    normalized_audio = audio_array
                
                # Save normalized segment to temporary file
                audio_temp = pyaudio.PyAudio()
                wf = wave.open(filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(sample_width)
                wf.setframerate(16000)
                wf.writeframes(normalized_audio.tobytes())
                wf.close()
                audio_temp.terminate()
                
                start_time = time.time()
                
                # Transcribe using faster-whisper with error handling
                try:
                    segments, info = self.model.transcribe(
                        filename,
                        language="en",
                        beam_size=5,
                        condition_on_previous_text=False,  # Reduce hallucinations
                        vad_filter=True,  # Use Whisper's built-in VAD
                        vad_parameters=dict(threshold=0.5, min_speech_duration_ms=250)
                    )
                except Exception as trans_err:
                    self.audio_queue.task_done()
                    self.is_processing = False
                    continue
                
                # Collect all text from segments
                transcribed_text = " ".join([segment.text for segment in segments]).strip()
                                
                self.segments_transcribed += 1
                
                # Delete temporary file immediately after transcription
                try:
                    os.unlink(filename)
                except Exception as del_err:
                    pass  # Silently ignore file deletion errors
                
                if transcribed_text:
                    self.full_transcription.append(transcribed_text)
                    # Schedule GUI update in main thread
                    self.window.after(0, self._update_text_area, transcribed_text)
                
                self.audio_queue.task_done()
                self.is_processing = False
                
            except queue.Empty:
                # No segments in queue, continue waiting
                self.is_processing = False
                continue
            except Exception as e:
                error_msg = f"\n\n[TRANSCRIBE] Error: {str(e)}\n"
                print(error_msg)
                print(f"[TRANSCRIBE] Full error details: {type(e).__name__}: {str(e)}")
                self.window.after(0, lambda msg=error_msg: self.text_area.insert(tk.END, msg))
                self.is_processing = False
                
                # Clean up temp file if error occurred
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
    
    def process_full_audio(self):
        """Process the complete recording for final, high-quality transcription"""
        if not self.full_audio_frames:
            self.status_label.config(text="No audio to process", fg="orange")
            return
        
        print("\n" + "="*60)
        print("PROCESSING FULL AUDIO RECORDING")
        print("="*60)
        
        # Combine all audio frames
        full_audio_data = b''.join(self.full_audio_frames)
        
        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filepath = temp_file.name
        temp_file.close()
        
        try:
            # Write complete audio to file
            with wave.open(temp_filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(full_audio_data)
            
            duration = len(full_audio_data) / (2 * 16000)  # 2 bytes per sample, 16000 Hz
            print(f"Total recording duration: {duration:.1f} seconds")
            print("Transcribing complete audio... (this may take a moment)")
            
            # Transcribe the COMPLETE audio with full context
            # No VAD filter for final - we want ALL speech
            segments, info = self.model.transcribe(
                temp_filepath,
                language="en",
                beam_size=5,
                condition_on_previous_text=True,  # Use context for better accuracy
                vad_filter=False  # Don't filter - transcribe everything
            )
            
            # Collect all segments - one per line for readability
            full_text_parts = []
            segment_count = 0
            for segment in segments:
                segment_count += 1
                text = segment.text.strip()
                print(f"  Segment {segment_count}: '{text}'")
                if text:
                    full_text_parts.append(text)
            
            print(f"Total segments found: {segment_count}")
            
            if not full_text_parts:
                final_transcription = "[No speech detected in recording]"
            else:
                # Join with newlines instead of spaces for better readability
                final_transcription = "\n".join(full_text_parts)
            
            # Store the final transcription for saving
            self.final_complete_transcription = final_transcription
                        
            # Update GUI with final transcription
            self.window.after(0, self._show_final_transcription, final_transcription)
            self.window.after(0, lambda: self.status_label.config(
                text="Complete! Final transcription shown above.", fg="green"
            ))
            
        except Exception as e:
            print(f"Error processing full audio: {e}")
            self.window.after(0, lambda: self.status_label.config(
                text="Error processing full audio", fg="red"
            ))
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_filepath)
            except:
                pass
    
    def _show_final_transcription(self, text):
        """Display the final complete transcription in the GUI"""
        self.text_area.insert(tk.END, "\n" + "="*50 + "\n")
        self.text_area.insert(tk.END, "FINAL COMPLETE TRANSCRIPTION:\n")
        self.text_area.insert(tk.END, "="*50 + "\n")
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.insert(tk.END, "="*50 + "\n")
        self.text_area.see(tk.END)
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TranscriptionApp()
    app.run()