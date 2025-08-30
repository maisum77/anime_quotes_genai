import os
import torch
import random
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class AnimeQuotesDataset(Dataset):
    """Custom dataset for anime quotes"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        
        for text in texts:
            encodings = tokenizer(text, truncation=True, padding='max_length', 
                                 max_length=max_length, return_tensors='pt')
            self.input_ids.append(encodings['input_ids'].squeeze())
            self.attention_masks.append(encodings['attention_mask'].squeeze())
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.input_ids[idx]
        }

class AnimeSpeechGenerator:
    def __init__(self, model_name='gpt2'):
        """Initialize the GPT-2 model and tokenizer"""
        print(f"Loading {model_name} model and tokenizer...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def clean_quotes(self, text):
        """Clean and prepare the quotes for training"""
        quotes = [line.strip() for line in text.split('\n') if line.strip()]
        
        cleaned_quotes = []
        for quote in quotes:
            # Remove surrounding quotes and fix encoding
            quote = quote.strip('"')
            quote = quote.replace('â€™', "'")
            quote = quote.replace('â€', '"')
            quote = quote.replace('â€¦', '...')
            quote = quote.replace('â€"', '-')
            cleaned_quotes.append(quote)
        
        return cleaned_quotes
    
    def prepare_training_data(self, quotes):
        """Prepare data for training by creating speech-like combinations"""
        training_texts = []
        
        # Individual quotes
        training_texts.extend(quotes)
        
        # Combine quotes into speeches
        speech_templates = [
            "Listen up! {0} Remember, {1} That's why {2}",
            "I'll tell you something important. {0} But here's the truth: {1} Never forget that {2}",
            "This is our moment! {0} We've come so far because {1} And now, {2}",
            "{0} You know why? Because {1} This is what it means to {2}",
            "Don't give up now! {0} Through all our battles, {1} Together, {2}"
        ]
        
        # Create synthetic speeches
        for _ in range(len(quotes) // 3):
            template = random.choice(speech_templates)
            selected_quotes = random.sample(quotes, min(3, len(quotes)))
            if len(selected_quotes) >= 3:
                speech = template.format(*selected_quotes[:3])
                training_texts.append(speech)
        
        # Create connected pairs
        for i in range(0, len(quotes)-1, 2):
            connected = f"{quotes[i]} That's why {quotes[i+1]}"
            training_texts.append(connected)
        
        return training_texts
    
    def fine_tune(self, quotes, output_dir='./anime_speech_model', epochs=3):
        """Fine-tune GPT-2 on anime quotes"""
        print("\nPreparing training data...")
        training_texts = self.prepare_training_data(quotes)
        print(f"Created {len(training_texts)} training samples")
        
        # Create dataset
        dataset = AnimeQuotesDataset(training_texts, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=100,
            logging_dir='./logs',
            warmup_steps=100,
            dataloader_drop_last=True,
            disable_tqdm=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )
        
        # Train
        print(f"\nStarting fine-tuning for {epochs} epochs...")
        print("This may take a few minutes...")
        trainer.train()
        
        # Save the model
        print(f"\nSaving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print("Fine-tuning complete!")
        return self.model
    
    def load_fine_tuned_model(self, model_path='./anime_speech_model'):
        """Load a fine-tuned model"""
        if os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            print("Fine-tuned model loaded!")
            return True
        return False
    
    def generate_speech(self, speech_type='motivational', temperature=0.8, max_length=200):
        """Generate an anime-style speech"""
        
        # Speech prompts based on type
        prompts = {
            'motivational': [
                "Listen everyone! True strength",
                "Never give up! The path to victory",
                "I'll show you what it means to",
                "This is why we fight! Because",
                "Stand up and keep moving forward!"
            ],
            'battle': [
                "This battle isn't over!",
                "I won't let you win because",
                "My power comes from",
                "You think you've won? But",
                "This is my true strength!"
            ],
            'friendship': [
                "We're not alone because",
                "My friends are my power and",
                "Together we can overcome",
                "The bonds we share will",
                "You're all precious to me because"
            ],
            'determination': [
                "I made a promise to",
                "No matter what happens, I will",
                "Even if I fall, I'll keep",
                "My dream is to become",
                "Nothing will stop me from"
            ],
            'villain': [
                "You fools don't understand that",
                "Power is everything and",
                "The weak deserve to",
                "This world will bow before",
                "Your hope is meaningless because"
            ]
        }
        
        # Select prompt
        prompt = random.choice(prompts.get(speech_type, prompts['motivational']))
        
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate with parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add dramatic ending if not present
        endings = [
            " That's what makes us strong!",
            " And that's why we'll never lose!",
            " This is our path to victory!",
            " Believe it!",
            " That's my ninja way!",
            " Together, we're unstoppable!"
        ]
        
        if not generated_text.endswith(('!', '.', '?')):
            generated_text += random.choice(endings)
        
        return generated_text
    
    def generate_dialogue(self, characters=['Hero', 'Rival'], num_exchanges=3, temperature=0.8):
        """Generate a dialogue between characters"""
        dialogue = []
        
        character_prompts = {
            'Hero': ["I'll never give up", "My friends give me strength", "I'll protect everyone"],
            'Rival': ["You're still weak", "Power is everything", "You can't defeat me"],
            'Mentor': ["Listen carefully", "True strength comes from", "You must understand"],
            'Friend': ["We're in this together", "I believe in you", "Don't give up"]
        }
        
        for i in range(num_exchanges):
            for char in characters:
                # Get character-specific prompt or use generic
                if char in character_prompts:
                    prompt = random.choice(character_prompts[char])
                else:
                    prompt = "I have something to say"
                
                # Generate speech
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=100,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95
                    )
                
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                dialogue.append(f"**{char}**: {text}")
        
        return "\n\n".join(dialogue)

def main():
    """Main function to run the anime speech generator"""
    
    print("=" * 60)
    print("ANIME SPEECH GENERATOR - GPT-2 EDITION")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ No GPU detected, using CPU (training will be slower)")
    
    # Initialize generator
    generator = AnimeSpeechGenerator('gpt2')  # Can use 'gpt2-medium' for better quality
    
    # Check for existing fine-tuned model
    model_exists = generator.load_fine_tuned_model()
    
    # Menu
    print("\n" + "=" * 60)
    print("OPTIONS:")
    print("1. Fine-tune on anime quotes (recommended first)")
    print("2. Generate motivational speech")
    print("3. Generate battle speech")
    print("4. Generate friendship speech")
    print("5. Generate villain monologue")
    print("6. Generate character dialogue")
    print("7. Custom prompt generation")
    print("8. Batch generate multiple speeches")
    print("0. Exit")
    
    while True:
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == '0':
            print("Goodbye, and never give up on your dreams!")
            break
        
        elif choice == '1':
            # Fine-tuning
            try:
                with open('data.txt', 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                quotes = generator.clean_quotes(raw_text)
                print(f"\nFound {len(quotes)} quotes for training")
                
                epochs = input("Enter number of epochs (1-5, default 3): ").strip()
                epochs = int(epochs) if epochs.isdigit() and 1 <= int(epochs) <= 5 else 3
                
                generator.fine_tune(quotes, epochs=epochs)
                model_exists = True
                
            except FileNotFoundError:
                print("\nError: 'data.txt' not found!")
                print("Creating sample data.txt with anime quotes...")
                
                sample_quotes = [
                    '"Never give up on your dreams!"',
                    '"True strength comes from protecting others."',
                    '"The bonds of friendship are unbreakable!"',
                    '"I will become stronger, no matter what!"',
                    '"Pain makes us stronger."',
                    '"Believe in yourself!"',
                    '"Together, we can overcome anything!"',
                    '"This is my ninja way!"',
                    '"I\'ll protect my friends!"',
                    '"Power comes from the heart!"'
                ]
                
                with open('data.txt', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sample_quotes))
                print("Sample data.txt created. Run option 1 again to train.")
        
        elif choice in ['2', '3', '4', '5']:
            # Generate speeches
            if not model_exists:
                print("\n⚠ Using base GPT-2. For better anime-style results, fine-tune first (Option 1)")
            
            speech_types = {
                '2': 'motivational',
                '3': 'battle',
                '4': 'friendship',
                '5': 'villain'
            }
            
            speech_type = speech_types[choice]
            
            temp = input("Temperature (0.5-1.0, default 0.8): ").strip()
            try:
                temp = float(temp)
                temp = max(0.5, min(1.0, temp))
            except:
                temp = 0.8
            
            print(f"\n{'='*60}")
            print(f"GENERATING {speech_type.upper()} SPEECH...")
            print(f"{'='*60}\n")
            
            speech = generator.generate_speech(speech_type, temperature=temp)
            print(speech)
            
            # Save option
            save = input("\nSave speech? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"gpt2_speech_{speech_type}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(speech)
                print(f"Saved to {filename}")
        
        elif choice == '6':
            # Generate dialogue
            print("\nCharacter options: Hero, Rival, Mentor, Friend")
            chars = input("Enter characters (comma-separated, or press Enter for Hero,Rival): ").strip()
            
            if chars:
                characters = [c.strip() for c in chars.split(',')]
            else:
                characters = ['Hero', 'Rival']
            
            num_ex = input("Number of exchanges (1-5, default 3): ").strip()
            num_ex = int(num_ex) if num_ex.isdigit() and 1 <= int(num_ex) <= 5 else 3
            
            print(f"\n{'='*60}")
            print("GENERATING DIALOGUE...")
            print(f"{'='*60}\n")
            
            dialogue = generator.generate_dialogue(characters, num_ex)
            print(dialogue)
            
            save = input("\nSave dialogue? (y/n): ").strip().lower()
            if save == 'y':
                with open("gpt2_dialogue.txt", 'w', encoding='utf-8') as f:
                    f.write(dialogue)
                print("Saved to gpt2_dialogue.txt")
        
        elif choice == '7':
            # Custom prompt
            prompt = input("Enter your custom prompt: ").strip()
            if prompt:
                temp = input("Temperature (0.5-1.0, default 0.8): ").strip()
                try:
                    temp = float(temp)
                    temp = max(0.5, min(1.0, temp))
                except:
                    temp = 0.8
                
                inputs = generator.tokenizer.encode(prompt, return_tensors='pt').to(generator.device)
                
                with torch.no_grad():
                    outputs = generator.model.generate(
                        inputs,
                        max_length=200,
                        temperature=temp,
                        pad_token_id=generator.tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95
                    )
                
                generated = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n{'='*60}")
                print("GENERATED TEXT:")
                print(f"{'='*60}\n")
                print(generated)
        
        elif choice == '8':
            # Batch generate
            num = input("How many speeches to generate? (1-10, default 5): ").strip()
            num = int(num) if num.isdigit() and 1 <= int(num) <= 10 else 5
            
            speeches = []
            types = ['motivational', 'battle', 'friendship', 'determination']
            
            print(f"\nGenerating {num} speeches...")
            for i in range(num):
                speech_type = random.choice(types)
                speech = generator.generate_speech(speech_type, temperature=0.8)
                speeches.append(f"SPEECH {i+1} ({speech_type.upper()}):\n{speech}\n")
            
            print("\n" + "="*60)
            for speech in speeches:
                print(speech)
                print("-"*40)
            
            save = input("\nSave all speeches? (y/n): ").strip().lower()
            if save == 'y':
                with open("gpt2_batch_speeches.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(speeches))
                print("Saved to gpt2_batch_speeches.txt")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import transformers
        import torch
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
        print("Packages installed! Please run the script again.")
        sys.exit()
    
    main()