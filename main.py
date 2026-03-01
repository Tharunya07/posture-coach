import asyncio
from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, User, Runner
from vision_agents.plugins import getstream, gemini, ultralytics

load_dotenv()

async def create_agent(**kwargs) -> Agent:
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Posture Coach", id="agent"),
        instructions="""You are a real-time posture coach watching via webcam.
        You have YOLO skeleton data showing the user's pose keypoints.
        Every time you are asked to check posture, look at the skeleton and give ONE short coaching tip.
        Focus on: head position, shoulder alignment, spine straightness.
        Be direct and specific. Max 1 sentence. Example: "Chin up, your head is drooping forward."
        If posture looks good, say "Good posture, keep it up." """,
        llm=gemini.Realtime(fps=5),
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")
        ],
    )

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        # Wait for participant to join
        await asyncio.sleep(5)
        
        # Keep pinging the agent every 8 seconds to trigger a response
        while True:
            await agent.simple_response("Check the user's posture right now and give feedback.")
            await asyncio.sleep(8)

if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()