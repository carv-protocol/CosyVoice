import sys
import os
import torch
import torchaudio
from pathlib import Path

# Add necessary paths
sys.path.append('third_party/Matcha-TTS')
# Update this path if needed
sys.path.append('/path/to/CosyVoice')  

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class CosyVoice2Manager:
    def __init__(self, model_path, use_jit=False, use_trt=False, fp16=False, use_flow_cache=False):
        """Initialize CosyVoice2 model with specified parameters"""
        print(f"Initializing CosyVoice2 from: {model_path}")
        
        try:
            # Initialize with text_frontend=False to avoid ttsfrd dependency
            self.model = CosyVoice2(
                model_path,
                load_jit=use_jit,
                load_trt=use_trt,
                fp16=fp16,
                use_flow_cache=use_flow_cache
            )
            self.sample_rate = self.model.sample_rate
            print(f"Model initialized successfully. Sample rate: {self.sample_rate}")
            
            # Check for existing speakers
            self.list_speakers()
        except Exception as e:
            print(f"Error initializing CosyVoice2: {e}")
            raise
    
    def list_speakers(self):
        """List and count available speaker profiles"""
        if hasattr(self.model, 'spk_dict') and self.model.spk_dict:
            spk_count = len(self.model.spk_dict)
            print(f"Number of speaker profiles: {spk_count}")
            
            print("\nAvailable speakers:")
            for idx, spk_id in enumerate(self.model.spk_dict.keys()):
                print(f"{idx+1}. {spk_id}")
            
            return spk_count, list(self.model.spk_dict.keys())
        else:
            print("No speaker profiles found or speaker dictionary is empty")
            return 0, []
    
    def add_speaker(self, name, prompt_audio_path, prompt_text=""):
        """Add a new speaker from audio prompt"""
        try:
            prompt_speech = load_wav(prompt_audio_path, 16000)  # Load at 16kHz
            success = self.model.add_zero_shot_spk(prompt_text, prompt_speech, name)
            
            if success:
                print(f"Speaker '{name}' added successfully")
                # Save speaker info to persist it
                self.save_speakers()
                return True
            else:
                print(f"Failed to add speaker '{name}'")
                return False
        except Exception as e:
            print(f"Error adding speaker: {e}")
            return False
    
    def save_speakers(self):
        """Save speaker profiles for persistence"""
        try:
            if hasattr(self.model, 'save_spkinfo'):
                self.model.save_spkinfo()
                print("Speaker profiles saved successfully")
                return True
            else:
                print("Warning: save_spkinfo method not found")
                return False
        except Exception as e:
            print(f"Error saving speaker info: {e}")
            return False
    
    def generate_speech(self, text, speaker_id=None, prompt_audio_path=None, 
                       prompt_text="", output_dir="output", stream=False):
        """Generate speech using a saved speaker or prompt audio"""
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        # Prepare prompt speech if needed
        prompt_speech = None
        if speaker_id is None and prompt_audio_path:
            prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # Generate speech
        try:
            if speaker_id:
                # Use a saved speaker profile
                print(f"Generating speech using saved speaker: {speaker_id}")
                generator = self.model.inference_zero_shot(
                    text, "", "", zero_shot_spk_id=speaker_id, stream=stream
                )
            else:
                # Use prompt audio directly
                print("Generating speech using prompt audio")
                generator = self.model.inference_zero_shot(
                    text, prompt_text, prompt_speech, stream=stream
                )
            
            # Save generated audio chunks
            for i, result in enumerate(generator):
                output_path = os.path.join(output_dir, f"speech_{i}.wav")
                torchaudio.save(
                    output_path, 
                    result['tts_speech'], 
                    self.sample_rate
                )
                output_files.append(output_path)
                print(f"Generated audio saved to: {output_path}")
            
            return output_files
        except Exception as e:
            print(f"Error generating speech: {e}")
            return []
    
    def generate_speech_with_instruction(self, text, instruction, speaker_id=None, 
                                        prompt_audio_path=None, output_dir="output", stream=False):
        """Generate speech with specific instructions for tone, emotion, etc."""
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        # Prepare prompt speech if needed
        prompt_speech = None
        if prompt_audio_path:
            prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # Generate speech with instruction
        try:
            print(f"Generating speech with instruction: '{instruction}'")
            
            # Check if the model has the instruction method
            if hasattr(self.model, 'inference_instruct2'):
                generator = self.model.inference_instruct2(
                    text, instruction, prompt_speech, stream=stream
                )
                
                # Save generated audio chunks
                for i, result in enumerate(generator):
                    output_path = os.path.join(output_dir, f"instruct_{i}.wav")
                    torchaudio.save(
                        output_path, 
                        result['tts_speech'], 
                        self.sample_rate
                    )
                    output_files.append(output_path)
                    print(f"Generated audio saved to: {output_path}")
                
                return output_files
            else:
                print("Error: This model does not support instruction-based generation")
                return []
        except Exception as e:
            print(f"Error generating speech with instruction: {e}")
            return []
    
    def generate_with_fine_control(self, text, prompt_audio_path=None, output_dir="output", stream=False):
        """Generate speech with fine-grained control using control tags in the text"""
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        # Prepare prompt speech if needed
        prompt_speech = None
        if prompt_audio_path:
            prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # Generate speech with fine-grained control
        try:
            print("Generating speech with fine-grained control (using control tags in text)")
            
            # Check if the model has the cross_lingual method (used for fine control)
            if hasattr(self.model, 'inference_cross_lingual'):
                generator = self.model.inference_cross_lingual(
                    text, prompt_speech, stream=stream
                )
                
                # Save generated audio chunks
                for i, result in enumerate(generator):
                    output_path = os.path.join(output_dir, f"fine_control_{i}.wav")
                    torchaudio.save(
                        output_path, 
                        result['tts_speech'], 
                        self.sample_rate
                    )
                    output_files.append(output_path)
                    print(f"Generated audio saved to: {output_path}")
                
                return output_files
            else:
                print("Error: This model does not support fine-grained control generation")
                return []
        except Exception as e:
            print(f"Error generating speech with fine control: {e}")
            return []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CosyVoice2 Advanced Manager")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model directory")
    parser.add_argument("--action", type=str, 
                        choices=["list", "add", "generate", "instruct", "fine_control", "advanced"], 
                        required=True, help="Action to perform")
    parser.add_argument("--speaker_name", type=str, help="Name for new speaker")
    parser.add_argument("--prompt_audio", type=str, help="Path to prompt audio file")
    parser.add_argument("--prompt_text", type=str, default="", help="Text corresponding to prompt audio")
    parser.add_argument("--text", type=str, help="Text to synthesize (can include control tags)")
    parser.add_argument("--instruction", type=str, 
                       help="Instruction for voice (e.g., 'speak sadly', 'use British accent')")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--speaker_id", type=str, help="ID of saved speaker to use")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    
    args = parser.parse_args()
    
    # Initialize the manager
    manager = CosyVoice2Manager(args.model_path)
    
    # Perform the requested action
    if args.action == "list":
        manager.list_speakers()
    
    elif args.action == "add":
        if not args.speaker_name or not args.prompt_audio:
            print("Error: speaker_name and prompt_audio are required for add action")
            return
        
        manager.add_speaker(args.speaker_name, args.prompt_audio, args.prompt_text)
    
    elif args.action == "generate":
        if not args.text:
            print("Error: text is required for generate action")
            return
        
        if not args.speaker_id and not args.prompt_audio:
            print("Error: either speaker_id or prompt_audio is required for generate action")
            return
        
        manager.generate_speech(
            args.text,
            speaker_id=args.speaker_id,
            prompt_audio_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            output_dir=args.output_dir,
            stream=args.stream
        )
    
    elif args.action == "instruct":
        if not args.text or not args.instruction:
            print("Error: text and instruction are required for instruct action")
            return
        
        manager.generate_speech_with_instruction(
            args.text,
            args.instruction,
            speaker_id=args.speaker_id,
            prompt_audio_path=args.prompt_audio,
            output_dir=args.output_dir,
            stream=args.stream
        )
    
    elif args.action == "fine_control":
        if not args.text:
            print("Error: text is required for fine_control action")
            return
        
        if not args.prompt_audio:
            print("Error: prompt_audio is required for fine_control action")
            return
        
        manager.generate_with_fine_control(
            args.text,
            prompt_audio_path=args.prompt_audio,
            output_dir=args.output_dir,
            stream=args.stream
        )


if __name__ == "__main__":
    main()