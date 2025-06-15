import streamlit as st
import json
import csv
import io
import re
from typing import List, Dict, Optional
import PyPDF2
import os
from datetime import datetime
import requests
import time
import google.generativeai as genai

class FlashcardGenerator:
    def __init__(self, api_key: str = None, model_type: str = "gemini"):
        """Initialize the flashcard generator with API key and model type."""
        self.model_type = model_type
        self.api_key = api_key
        
        # Configure Gemini
        if api_key and model_type == "gemini":
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
            
        self.subjects = {
            "General": "general knowledge",
            "Biology": "biological concepts, processes, and terminology",
            "History": "historical events, dates, figures, and concepts",
            "Computer Science": "programming concepts, algorithms, and technical terminology",
            "Mathematics": "mathematical concepts, formulas, and problem-solving",
            "Physics": "physical laws, concepts, and equations",
            "Chemistry": "chemical processes, reactions, and compounds",
            "Literature": "literary works, authors, and literary devices"
        }
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def clean_and_structure_text(self, text: str) -> str:
        """Clean and structure the input text for better processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        return '\n\n'.join(paragraphs)
    
    def generate_flashcards_prompt(self, content: str, subject: str, num_cards: int = 15) -> str:
        """Generate the prompt for LLM to create flashcards."""
        subject_context = self.subjects.get(subject, "general knowledge")
        
        prompt = f"""
You are an expert educator creating flashcards for {subject_context}. 

Content to analyze:
{content}

Instructions:
1. Create exactly {num_cards} high-quality flashcards from this content
2. Each flashcard should have a clear, specific question and a complete, accurate answer
3. Focus on key concepts, definitions, important facts, and relationships
4. Vary question types: definitions, explanations, comparisons, examples, applications
5. Ensure answers are self-contained and don't require the original text
6. If possible, identify topic sections and group related flashcards

Format your response as a JSON object with this exact structure:
{{
    "flashcards": [
        {{
            "id": 1,
            "topic": "Main topic or section name",
            "question": "Clear, specific question",
            "answer": "Complete, accurate answer",
            "difficulty": "Easy|Medium|Hard"
        }}
    ]
}}

Make sure the JSON is valid and properly formatted. Focus on the most important and testable information from the content.
"""
        return prompt
    
    def call_gemini_api(self, prompt: str) -> Optional[Dict]:
        """Call Google Gemini API for flashcard generation."""
        if not self.gemini_model:
            return None
            
        try:
            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    # Try parsing the entire response as JSON
                    return json.loads(response.text)
            
            return None
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing Gemini response: {str(e)}")
            return None
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                st.error("🚨 Gemini API quota exceeded. Switching to demo mode...")
            elif "invalid" in error_msg.lower() or "auth" in error_msg.lower():
                st.error("❌ Invalid Gemini API key. Please check your key.")
            else:
                st.error(f"Error calling Gemini: {error_msg}")
            return None
    def call_huggingface_api(self, prompt: str) -> Optional[Dict]:
        """Call Hugging Face Inference API for free flashcard generation."""
        try:
            # Using a free model that works well for text generation
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            # Simplified prompt for better results with smaller models
            simplified_prompt = f"""
Create 10 flashcards from this content about {self.subjects.get('General', 'general knowledge')}:

{prompt[:1000]}...

Format as JSON:
{{"flashcards": [{{"id": 1, "topic": "Topic", "question": "Question?", "answer": "Answer", "difficulty": "Medium"}}]}}
"""
            
            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"}
            
            response = requests.post(API_URL, 
                                   headers=headers, 
                                   json={"inputs": simplified_prompt, "parameters": {"max_length": 1000}})
            
            if response.status_code == 200:
                result = response.json()
                # Try to extract JSON from response
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
            
            return None
            
        except Exception as e:
            st.error(f"Error calling Hugging Face API: {str(e)}")
            return None

    def generate_mock_flashcards(self, content: str, subject: str = "General", num_cards: int = 15) -> List[Dict]:
        """Generate mock flashcards for demonstration when no API is available."""
        # This is a fallback method that creates sample flashcards based on content analysis
        # In a real implementation, you'd want to use a local model or different API
        
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        mock_cards = []
        for i in range(min(num_cards, len(sentences))):
            sentence = sentences[i]
            
            # Simple question generation based on sentence content
            if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were']):
                question = f"What {sentence.lower().replace('is', 'is').replace('are', 'are')}?"
            elif any(word in sentence.lower() for word in ['process', 'method', 'way']):
                question = f"Explain the process mentioned: {sentence[:50]}..."
            else:
                question = f"What is the main concept in: {sentence[:50]}...?"
            
            mock_cards.append({
                'id': i + 1,
                'topic': subject,
                'question': question[:100] + "?" if not question.endswith('?') else question[:100],
                'answer': sentence.strip(),
                'difficulty': ['Easy', 'Medium', 'Hard'][i % 3]
            })
        
        return mock_cards[:num_cards]
    def call_llm(self, prompt: str) -> Optional[Dict]:
        """Call the LLM API to generate flashcards with fallback options."""
        if self.model_type == "gemini" and self.gemini_model:
            return self.call_gemini_api(prompt)
        elif self.model_type == "huggingface":
            return self.call_huggingface_api(prompt)
        else:
            st.warning("⚠️ No API configured. Using demo mode.")
            return None
    
    def generate_flashcards(self, content: str, subject: str = "General", num_cards: int = 15) -> List[Dict]:
        """Generate flashcards from the given content with fallback to demo mode."""
        cleaned_content = self.clean_and_structure_text(content)
        prompt = self.generate_flashcards_prompt(cleaned_content, subject, num_cards)
        
        result = self.call_llm(prompt)
        if result and 'flashcards' in result:
            return result['flashcards']
        else:
            # Fallback to mock generation for demonstration
            st.info("🎯 Generating demo flashcards (API not available). This shows the app functionality!")
            return self.generate_mock_flashcards(content, subject, num_cards)
    
    def export_to_csv(self, flashcards: List[Dict]) -> str:
        """Export flashcards to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Topic', 'Question', 'Answer', 'Difficulty'])
        
        for card in flashcards:
            writer.writerow([
                card.get('id', ''),
                card.get('topic', ''),
                card.get('question', ''),
                card.get('answer', ''),
                card.get('difficulty', 'Medium')
            ])
        
        return output.getvalue()
    
    def export_to_json(self, flashcards: List[Dict]) -> str:
        """Export flashcards to JSON format."""
        return json.dumps({
            "generated_at": datetime.now().isoformat(),
            "total_cards": len(flashcards),
            "flashcards": flashcards
        }, indent=2)
    
    def export_to_anki(self, flashcards: List[Dict]) -> str:
        """Export flashcards to Anki-compatible format."""
        output = []
        for card in flashcards:
            # Anki format: Question<tab>Answer<tab>Tags
            tags = f"{card.get('topic', '').replace(' ', '_')} {card.get('difficulty', 'Medium')}"
            line = f"{card.get('question', '')}\t{card.get('answer', '')}\t{tags}"
            output.append(line)
        
        return '\n'.join(output)

def main():
    st.set_page_config(
        page_title="LLM Flashcard Generator",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 LLM-Powered Flashcard Generator")
    st.markdown("Transform your educational content into effective flashcards using AI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_option = st.selectbox(
            "Choose AI Model",
            ["Google Gemini (Free)", "Demo Mode (No API needed)", "Hugging Face (Free)"],
            help="Select the AI model to use for generation"
        )
        
        api_key = None
        model_type = "demo"
        
        if model_option == "Google Gemini (Free)":
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Enter your Google AI Studio API key (free with generous limits)"
            )
            model_type = "gemini" if api_key else "demo"
            
            if not api_key:
                st.info("💡 Get your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)")
            
        elif model_option == "Hugging Face (Free)":
            api_key = st.text_input(
                "Hugging Face API Key (Optional)",
                type="password",
                help="Enter your HF token for better rate limits (optional)"
            )
            model_type = "huggingface"
            
        # Subject selection
        subject = st.selectbox(
            "Subject Area",
            options=list(FlashcardGenerator().subjects.keys()),
            help="Select the subject to optimize flashcard generation"
        )
        
        # Number of flashcards
        num_cards = st.slider(
            "Number of Flashcards",
            min_value=5,
            max_value=25,
            value=15,
            help="Choose how many flashcards to generate"
        )
        
        # API Status
        if model_option == "Demo Mode (No API needed)":
            st.success("✅ Demo mode active - No API key needed!")
        elif model_option == "Google Gemini (Free)":
            if api_key:
                st.success("✅ Gemini API configured - Free tier with generous limits!")
            else:
                st.warning("⚠️ Add API key or use Demo Mode")
        elif model_option == "Hugging Face (Free)":
            st.info("🤗 Using free Hugging Face models")
    
    # Initialize the generator
    generator = FlashcardGenerator(api_key, model_type)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Content")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload File"]
        )
        
        content = ""
        
        if input_method == "Paste Text":
            content = st.text_area(
                "Paste your educational content here:",
                height=300,
                placeholder="Enter lecture notes, textbook excerpts, or any educational material..."
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'pdf'],
                help="Upload a .txt or .pdf file containing your educational content"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    content = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    content = generator.extract_text_from_pdf(uploaded_file)
                
                if content:
                    st.success(f"File uploaded successfully! Content length: {len(content)} characters")
                    with st.expander("Preview content"):
                        st.text(content[:500] + "..." if len(content) > 500 else content)
    
    with col2:
        st.header("🎯 Generated Flashcards")
        
        if st.button("🚀 Generate Flashcards", type="primary", disabled=not content):
            if len(content.strip()) < 50:
                st.warning("Please provide more content (at least 50 characters) for better flashcard generation.")
            else:
                with st.spinner("Generating flashcards... This may take a few moments."):
                    flashcards = generator.generate_flashcards(content, subject, num_cards)
                
                if flashcards:
                    st.success(f"Generated {len(flashcards)} flashcards!")
                    
                    # Store flashcards in session state
                    st.session_state.flashcards = flashcards
                    
                    # Display flashcards
                    st.subheader("Preview Flashcards")
                    
                    # Group by topic
                    topics = {}
                    for card in flashcards:
                        topic = card.get('topic', 'General')
                        if topic not in topics:
                            topics[topic] = []
                        topics[topic].append(card)
                    
                    for topic, cards in topics.items():
                        st.markdown(f"### 📚 {topic}")
                        for i, card in enumerate(cards, 1):
                            with st.expander(f"Card {card.get('id', i)}: {card.get('question', '')[:50]}..."):
                                st.markdown(f"**Question:** {card.get('question', '')}")
                                st.markdown(f"**Answer:** {card.get('answer', '')}")
                                st.markdown(f"**Difficulty:** {card.get('difficulty', 'Medium')}")
                
                else:
                    st.error("Failed to generate flashcards. Please check your API key and try again.")
    
    # Export section
    if 'flashcards' in st.session_state and st.session_state.flashcards:
        st.header("📤 Export Flashcards")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download CSV"):
                csv_data = generator.export_to_csv(st.session_state.flashcards)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"flashcards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download JSON"):
                json_data = generator.export_to_json(st.session_state.flashcards)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"flashcards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("Download Anki Format"):
                anki_data = generator.export_to_anki(st.session_state.flashcards)
                st.download_button(
                    label="🎴 Download for Anki",
                    data=anki_data,
                    file_name=f"flashcards_anki_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and OpenAI GPT | [View on GitHub](https://github.com/your-username/flashcard-generator)")

if __name__ == "__main__":
    main()