import os
from crewai import Agent, Task, Crew, LLM

# Set your Google API 

# Configure Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.environ['GOOGLE_API_KEY']
)

# Create a simple agent using Gemini
test_agent = Agent(
    role='Test Agent',
    goal='Test if Gemini LLM is working with CrewAI',
    backstory='A simple agent for testing Google Gemini integration.',
    llm=gemini_llm,
    verbose=True
)

# Create a simple task to test Gemini
test_task = Task(
    description='Say "Hello from Google Gemini!" and tell me one interesting fact about AI.',
    agent=test_agent,
    expected_output='A greeting from Gemini with an AI fact'
)

# Create the crew
crew = Crew(
    agents=[test_agent],
    tasks=[test_task],
    verbose=True
)

# Run the test
if __name__ == "__main__":
    print("Testing CrewAI with Google Gemini...")
    result = crew.kickoff()
    print("\n" + "="*40)
    print("GEMINI LLM TEST RESULT:")
    print("="*40)
    print(result)
    print("\nGoogle Gemini + CrewAI is working!")