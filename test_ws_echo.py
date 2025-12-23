import asyncio
import aiohttp
import json
import time
import wave
import struct

SERVER_URL = "ws://localhost:8000/v1/audio/speech/stream/ws"
OUTPUT_FILE = "test_output_ws.wav"
SAMPLE_RATE = 44100  # Match Server Config
# voice = "maya_long_ref"
# voice = "golum"
voice = "little_dude"
# voice = "expresso_02_ex03-ex01_calm_005"
# voice = "maya_voices"
# text = "(laughs) This is so tedious. I can barely stay awake through this."


# # text = "(singing) Happy birthday to you. Hope you have an amazing day today?"
# # text = "(singing) Happy birthday to you."
# # text = "(singing) Hope you have an amazing day today."

text = "Hello (laughs) This is a test of the Echo TTS streaming WebSocket. How does it sound?"
text = "That's hilarious! (laughs) I can't believe that actually happened to you."
text = "I'm so happy for you! (laughs) That's the best news I've heard all week."

# text = "Hey, I think someone's at the door."
text = "(whispers) Should I check who it is?"

# text = "(sighs) I've been trying to solve this problem all day."
# # text = "(sighs) I tried so hard to prevent it."
text = "(sighs) Everything feels different now."

text = "(sobbing) I just can't handle this anymore."
# # text = "(sobbing) Everything is falling apart."
# # text = "(sobbing) Why did this have to happen?"

# # text = "Wait, what (gasps) ? I had no idea that was even possible!"
# text = "(gasps) Oh my goodness! I wasn't expecting that at all."

# # text = "(yawns) Sorry, I'm a bit tired."
# text = "(yawns) What were you saying about the meeting?"

# text = "Excuse me (coughs). Now, where were we in our conversation? I thought we were talking about the weather."
# # text = "I'm not feeling well today (coughs) ."

# # text = "(shouts) Watch out! Get out of the way right now!"

# # ===== Emotion Tags =====
# # Happy - Cheerful, upbeat tone (Good news, greetings)
# # text = "Good morning! I have some wonderful news to share with you today."
# # text = "(happy) Welcome back! It's so great to see you again."
# # text = "(happy) Everything went perfectly! The results exceeded all our expectations."

# # Sad - Melancholic, downcast (Sympathy, bad news)
# # text = "I'm so sorry for your loss. Please know that I'm here for you."
# # text = "(sad) Unfortunately, we didn't get the results we were hoping for this time."
# # text = "Is there anything I can do to help? I know this must be difficult."

# # Angry - Frustrated, aggressive (Complaints, warnings)
# # text = "(angry) I can't believe you did that!"
# # text = "(angry) This is completely unacceptable."
# # text = "(angry) This is the third time this has happened. Something needs to change now."
# # text = "(angry) You were warned about this repeatedly, and you still ignored it."

# # Excited - Energetic, enthusiastic (Announcements, celebrations)
# # text = "(excited) Wow, that's incredible! You really did an amazing job on this project."
# # text = "(excited) I can't wait to tell you what just happened! This is going to be huge."
# # text = "(excited) We did it! The launch was a massive success beyond our wildest dreams."

# # Calm - Peaceful, relaxed (Instructions, meditation)
# # text = "(calm) Take a deep breath and relax."
# # text = "(calm) Everything is going to be just fine."
# # text = "(calm) Let me walk you through the steps slowly and carefully."
# # text = "(calm) There's no need to rush."

# # Nervous - Anxious, uncertain (Disclaimers, apologies)
# # text = "(nervous) I'm not entirely sure if this is the right approach, to be honest."
# # text = "(nervous) I apologize if this causes any inconvenience. I hope you understand."
# # text = "(nervous) Um I think there might be a small problem we need to address."

# # Confident - Assertive, self-assured (Presentations, sales)
# # text = "(confident) I can guarantee you that this solution will solve all your problems."
# # text = "(confident) Trust me, I've done this hundreds of times. We've got this covered."
# # text = "This is the best product on the market, and the numbers prove it."

# # Surprised - Shocked, amazed (Reactions, discoveries)
# text = "(surprised) Really? That's not what I thought would happen at all!"
# # text = "Wait, what? I had no idea that was even possible!"
# # text = "(surprised) Oh my god"
# # text = "(surprised) I wasn't expecting that result in the slightest."

# # Satisfied - Content, pleased (Confirmations, reviews)
# text = "(satisfied) Perfect! That's exactly what I was looking for. Well done."
# text = "Yes, this meets all the requirements. I'm very pleased with the outcome."
# text = "(satisfied) Everything is in order now. Thank you for your excellent work."

# # Delighted - Very pleased, joyful (Celebrations, compliments)
# text = "(delighted) This is absolutely wonderful! You've truly outdone yourself this time."
# text = "(delighted) I'm so thrilled with how everything turned out! It's perfect."
# # text = "(delighted) What a beautiful surprise! This made my entire day so much better."

# # Scared - Frightened, fearful (Warnings, horror stories)
# text = "(scared) Did you hear that noise? I think there's something out there."
# text = "(scared) Please be careful! That area is extremely dangerous at night."
# text = "(scared) I don't want to go in there. Something feels very wrong about this."

# # Worried - Concerned, troubled (Concerns, questions)
# text = "(worried) I'm really concerned about the deadline. Can we actually finish on time?"
# text = "(worried) Are you sure everything is okay."
# text  = "(worried) You seem a bit off today."
# # text = "(worried) What if something goes wrong? We need to have a backup plan ready."

# # Upset - Disturbed, distressed (Complaints, problems)
# text = "(upset) This is really bothering me. Why does this keep happening over and over?"
# text = "I'm very disappointed with how this situation was handled."
# text = "(upset) Nothing is working the way it should. This is extremely frustrating."

# # Frustrated - Annoyed, exasperated (Technical issues, delays)
# # text = "(frustrated) This isn't working no matter what I try. I don't understand why."
# text = "(frustrated) We've been delayed again for the fifth time this month!"
# text = "(frustrated) Why is this so complicated?"

# # Depressed - Very sad, hopeless (Serious topics)
# text = "(depressed) I just don't see the point anymore. Everything feels meaningless."
# text = "(depressed) Nothing I do seems to make any difference. It's all hopeless."
# text = "(depressed) I can't find any motivation to keep going. It's just too hard."

# # Empathetic - Understanding, caring (Support, counseling)
# text = "(empathetic) I understand how difficult this must be for you right now."
# text = "(empathetic) Your feelings are completely valid." 
# # text = "Anyone would feel the same way."
# # text = "(empathetic) I hear you, and I want you to know that you're not alone in this."

# # Embarrassed - Ashamed, awkward (Apologies, mistakes)
# text = "(embarrassed) Oh no, I can't believe I just did that in front of everyone."
# text = "(embarrassed) I'm so sorry about the mistake. I feel terrible about this."
# # text = "(embarrassed) That was really awkward. I wish I could take it back."

# # Disgusted - Repelled, revolted (Negative reviews)
# text = "(disgusted) No way! I'm not going anywhere near that mess."
# # text = "(disgusted) That's absolutely revolting. How could anyone think this is acceptable?"
# # text = "(disgusted) This is the worst experience I've ever had. Completely unacceptable."

# # Moved - Emotionally touched (Heartfelt moments)
# text = "(moved) That was such a beautiful gesture. I'm truly touched by your kindness."
# text = "Your words mean more to me than you could ever know. Thank you."
# # text = "I never expected this. You've really made a difference in my life."

# # Proud - Accomplished, satisfied (Achievements, praise)
# text = "(proud) Look at what we've accomplished together! This is truly remarkable."
# text = "(proud) You should be very proud of yourself."
# # text = "(proud) We've come so far and overcome so many obstacles. Well done everyone."

# # Relaxed - At ease, casual (Casual conversation)
# text = "(relaxed) Hey, how's it going? Just taking it easy today, nothing too serious."
# # text = "(relaxed) No worries at all. We can handle that whenever you're ready."
# text = "(relaxed) Yeah, sounds good to me. Let's just see how things go."

# # Grateful - Thankful, appreciative (Thanks, appreciation)
# text = " Thank you so much for all your help. I really appreciate everything you've done."
# # text = "(grateful) I'm so thankful to have you on the team. You make such a difference."
# # text = "(grateful) I can't thank you enough for your support during this difficult time."

# # Curious - Inquisitive, interested (Questions, exploration)
# # text = "(curious) That's interesting! How exactly does that work? I'd love to know more."
# text = "I wonder what would happen if we tried a different approach?"
# # text = "(curious) Tell me more about that. What made you think of this solution?"

# # Sarcastic - Ironic, mocking (Humor, criticism)
# text = "(sarcastic) Oh great, another meeting. Just what I needed to make my day complete."
# text = "(sarcastic) Wow, what a brilliant idea. I'm sure that will work out perfectly."
# # text = "(sarcastic) Yeah, because that worked so well the last three times we tried it."

# # Disdainful - Contemptuous, scornful (Criticism, rejection)
# text = "(disdainful) I have no interest in listening to such ridiculous proposals."
# text = " Your work is clearly beneath the standards we expect here."
# text = "(disdainful) That's the most absurd thing I've heard all week."

# # Unhappy - Discontent, dissatisfied (Complaints, feedback)
# text = " I'm not satisfied with how things are going right now."
# text = "This isn't what I expected at all. Something needs to change."
# # text = "(unhappy) I'm really not pleased with the current situation."

# # Anxious - Very worried, uneasy (Urgent matters)
# text = "(anxious) We need to handle this immediately! There's no time to waste."
# # text = "(anxious) I'm extremely worried about what might happen if we don't act now."
# # text = "(anxious) This is urgent! We have to make a decision right away."

# # Hysterical - Uncontrollably emotional (Extreme reactions)
# text = "(hysterical) I can't believe this is happening! This is absolutely insane!"
# text = "(hysterical) Everything is falling apart! What are we going to do?"
# # text = "(hysterical) This is a complete disaster! How could this happen to us?"

# # Indifferent - Uncaring, neutral (Neutral responses)
# text = "(indifferent) I don't really care either way. Whatever you decide is fine."
# # text = "(indifferent) It makes no difference to me. Do what you want."
# # text = "(indifferent) Sure, if that's what you think. I have no strong opinion."

# # Uncertain - Doubtful, unsure (Speculation, questions)
# text = "I'm not really sure if that's the right direction to take."
# # text = "(uncertain) Maybe that could work? I honestly don't know for certain."
# # text = "(uncertain) I'm having trouble deciding what the best approach would be."

# # Doubtful - Skeptical, questioning (Disbelief, questioning)
# text = "(doubtful) Are you sure that's going to work? I have my doubts."
# # text = "(doubtful) I'm not convinced this is the right solution to the problem."
# # text = "(doubtful) That seems unlikely to succeed, but I could be wrong."

# # Confused - Puzzled, perplexed (Clarification requests)
# text = "(confused) Wait, I don't understand. Can you explain that again?"
# text = "I'm completely lost. What exactly are you trying to say?"
# text = " This doesn't make any sense to me. Could you clarify?"

# # Disappointed - Let down, dissatisfied (Unmet expectations)
# text = "(disappointed) I really thought this would turn out better than it did."
# text = " This isn't what I was hoping for at all. I expected more."
# # text = "(disappointed) I'm let down by these results. We deserved better."

# # Regretful - Sorry, remorseful (Apologies, mistakes)
# # text = "(regretful) I wish I had made a different choice. I really regret this."
# text = "(regretful) Looking back, I should have handled that situation differently."
# # text = "(regretful) If only I had listened to you. I'm sorry I didn't."

# # Guilty - Culpable, responsible (Confessions, apologies)
# text = "(guilty) It's all my fault. I take full responsibility for what happened."
# text = "(guilty) I'm the one to blame for this mess. I should have known better."
# # text = "(guilty) I feel terrible about what I did. I knew it was wrong."

# # Ashamed - Deeply embarrassed (Serious mistakes)
# text = "(ashamed) I'm so ashamed of my behavior. I let everyone down."
# text = "(ashamed) I can't believe I did something so terrible. I'm deeply sorry."
# # text = "(ashamed) I'm mortified by what happened. I'll never forgive myself."

# # Jealous - Envious, resentful (Comparisons)
# text = "(jealous) Why does everyone always pay attention to them and not me?"
# text = "(jealous) I can't help but feel resentful when I see their success."
# text = "(jealous) It's not fair that they get all the recognition and praise."

# # Envious - Wanting what others have (Admiration with desire)
# text = "(envious) I wish I had what they have. They're so fortunate."
# # text = "(envious) Their life seems so perfect. I'd love to be in their position."
# # text = "(envious) I really admire what they've achieved. I want that too."

# # Hopeful - Optimistic about future (Future plans)
# text = "(hopeful) I believe things are going to get better very soon."
# text = "(hopeful) I'm optimistic that we'll find a solution to this problem."
# text = "(hopeful) There's still a chance this could work out in our favor."

# # Optimistic - Positive outlook (Encouragement)
# text = "(optimistic) Everything is going to work out perfectly. I'm sure of it."
# text = "(optimistic) This is a great opportunity! I know we'll succeed."
# # text = "(optimistic) Don't worry, the future looks bright and full of possibilities."

# # Pessimistic - Negative outlook (Warnings, doubts)
# text = "(pessimistic) I don't think this is going to work out well at all."
# # text = "(pessimistic) Things are probably going to get worse before they get better."
# # text = "(pessimistic) I have a bad feeling about this. It's likely to fail."

# # Nostalgic - Longing for the past (Memories, stories)
# text = " I remember when things were so much simpler back then."
# text = "(nostalgic) Those were the good old days. I miss them so much."
# text = "(nostalgic) It brings back such wonderful memories from years ago."

# # Lonely - Isolated, alone (Emotional content)
# text = "(lonely) I feel so alone right now. Nobody seems to understand me."
# # text = "(lonely) It's hard being by myself all the time with no one to talk to."
# # text = "(lonely) I wish I had someone here with me. The isolation is difficult."

# # Bored - Uninterested, weary (Disinterest)
# text = "This is so tedious. (laughs) I can barely stay awake through this."
# # text = "(bored) I'm completely uninterested in what's happening right now."
# # text = "(bored) Can we please do something else? This is incredibly dull."

# # Contemptuous - Showing contempt (Strong criticism)
# # text = "(contemptuous) Your incompetence is truly astounding and pathetic."
# # text = "(contemptuous) I have nothing but contempt for such lazy work."
# # text = "(contemptuous) This is beneath me. I refuse to waste my time on it."

# # Sympathetic - Showing sympathy (Condolences)
# # text = "(sympathetic) I'm so sorry you're going through this difficult time."
# # text = "(sympathetic) My heart goes out to you. This must be really hard."
# # text = "(sympathetic) I feel for you. Please let me know if there's anything I can do."

# # Compassionate - Showing deep care (Support, help)
# # text = "(compassionate) I truly care about your wellbeing and want to help you."
# # text = "(compassionate) Let me support you through this. You don't have to face it alone."
# # text = "(compassionate) Your pain matters to me. I'm here for you completely."

# # Determined - Resolved, decided (Goals, commitments)
# # text = "(determined) I will not give up until we achieve our goal. Nothing will stop me."
# # text = "(determined) I'm absolutely committed to making this work no matter what."
# # text = "(determined) We're going to succeed. I'm resolved to see this through."

# # Resigned - Accepting defeat (Giving up, acceptance)
# # text = "(resigned) I guess there's nothing more we can do. It's over."
# # text = "(resigned) I've accepted that this isn't going to work out. Time to move on."
# # text = "(resigned) Fine, you win. I'm done fighting this losing battle."



seed = 101  # Random seed for reproducibility
# Advanced sampler configuration (optional)
extra_body = {
    "chunking_enabled": False,
    # "speaker_kv_scale": 1.33,
    # "speaker_kv_min_t": 0.9,
    # "speaker_kv_max_layers": 24,
    # Uncomment to customize sampler settings:
    # "block_sizes": [32, 128, 480],
    # "num_steps": [8, 15, 20],
    # "cfg_scale_text": 3.0,
    # "cfg_scale_speaker": 8.0,
}

async def test_echo_ws():
    print(f"Connecting to {SERVER_URL}...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(SERVER_URL) as ws:
                print("Connected!")
                
                # Prepare a request
               
                segment_id = "test_seg_1"
                
                request = {
                    "input": text,
                    "voice": voice,
                    "segment_id": segment_id,
                    "seed": seed,
                    "extra_body": extra_body,
                    "continue": True
                }
                
                print(f"Sending request: {json.dumps(request, indent=2)}")
                await ws.send_json(request)
                
                # Variables to track performance
                start_time = time.perf_counter()
                first_chunk = True
                audio_data = bytearray()
                
                print("Waiting for response...")
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "start":
                            print(f"[Server] Started segment: {data.get('segment_id')}")
                        elif msg_type == "end":
                            print(f"[Server] Ended segment: {data.get('segment_id')}")
                            # Send close signal
                            close_req = {"continue": False}
                            print("Sending close signal...")
                            await ws.send_json(close_req)
                        elif "done" in data:
                            print("[Server] Done signal received. Closing.")
                            break
                        elif "error" in data:
                            print(f"[Server] Error: {data['error']}")
                            break
                        else:
                            print(f"[Server] Unknown JSON: {data}")
                            
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        if first_chunk:
                            ttfb = (time.perf_counter() - start_time) * 1000
                            print(f"⚡ TTFB: {ttfb:.2f} ms")
                            first_chunk = False
                        
                        chunk_len = len(msg.data)
                        if chunk_len % 2 != 0:
                            print(f"⚠️ Warning: Odd chunk size received: {chunk_len} bytes")
                        
                        # Debug: check first/last bytes of chunk
                        # print(f"Chunk: {chunk_len} bytes. Start: {msg.data[:4].hex()} End: {msg.data[-4:].hex()}")
                        
                        audio_data.extend(msg.data)
                    
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        print("WebSocket closed by server.")
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print("WebSocket error.")
                        break
                
                # Save audio to WAV
                if audio_data:
                    print(f"Saving {len(audio_data)} bytes to {OUTPUT_FILE}...")
                    try:
                        with wave.open(OUTPUT_FILE, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2) # 16-bit PCM
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(audio_data)
                        print(f"✅ Saved audio to {OUTPUT_FILE}")
                    except Exception as e:
                        print(f"❌ Failed to save audio: {e}")
                else:
                    print("❌ No audio data received.")
                    
        except aiohttp.ClientConnectorError as e:
            print(f"❌ Connection failed: {e}")
            print("Make sure the server is running: python server.py")

if __name__ == "__main__":
    asyncio.run(test_echo_ws())
