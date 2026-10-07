import os
import asyncio
from typing import Optional, List

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
import google.generativeai as genai

# ===== LOAD CONFIG =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "คุณคือผู้ช่วย AI ที่เป็นมิตร ตอบเป็นภาษาไทย สั้นกระชับ",
)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
TRIGGER_MODE = os.getenv("TRIGGER_MODE", "slash").lower()
ALLOWED_CHANNELS = {
    int(c.strip())
    for c in os.getenv("ALLOWED_CHANNELS", "").split(",")
    if c.strip().isdigit()
}

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise ValueError("❌ ตั้งค่า DISCORD_TOKEN และ GEMINI_API_KEY ใน .env ก่อน")

# ===== GEMINI SETUP =====
genai.configure(api_key=GEMINI_API_KEY)
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_output_tokens=2000,
    candidate_count=1,
)
safety_settings = {
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
}
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# ===== DISCORD SETUP =====
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)
tree = bot.tree

# ===== MEMORY STORE =====
chat_histories: dict[int, List[dict]] = {}


def get_history(channel_id: int) -> List[dict]:
    return chat_histories.setdefault(channel_id, [])


def add_history(channel_id: int, role: str, text: str):
    hist = get_history(channel_id)
    hist.append({"role": role, "parts": [text]})
    if len(hist) > MAX_HISTORY * 2:
        chat_histories[channel_id] = hist[-(MAX_HISTORY * 2) :]


def clear_history(channel_id: int):
    chat_histories.pop(channel_id, None)


def is_allowed_channel(channel: discord.abc.GuildChannel) -> bool:
    return not ALLOWED_CHANNELS or channel.id in ALLOWED_CHANNELS


# ===== HELPERS =====
async def send_long_message(destination, text: str, reference: Optional[discord.Message] = None):
    if not text:
        return
    chunks = [text[i : i + 1900] for i in range(0, len(text), 1900)]
    for i, chunk in enumerate(chunks):
        if i == 0 and reference:
            await reference.reply(chunk, mention_author=False)
        else:
            await destination.send(chunk)


async def process_ai_response(
    channel: discord.TextChannel,
    prompt: str,
    reference_msg: Optional[discord.Message] = None,
    attachments: Optional[List[discord.Attachment]] = None,
):
    if not is_allowed_channel(channel):
        return

    async with channel.typing():
        try:
            hist = get_history(channel.id)
            chat = model.start_chat(history=hist)

            content_parts = [prompt] if prompt else []
            if attachments:
                for att in attachments[:4]:
                    if att.content_type and att.content_type.startswith("image/"):
                        try:
                            image_bytes = await att.read()
                            content_parts.append({
                                "mime_type": att.content_type,
                                "data": image_bytes,
                            })
                        except Exception as exc:
                            print(f"⚠️ Image read failed: {exc}")

            if not content_parts:
                await channel.send(
                    "🤔 ไม่ได้มีข้อความหรือรูปภาพให้ประมวลผลครับ",
                    reference=reference_msg,
                    mention_author=False,
                )
                return

            response = await asyncio.to_thread(chat.send_message, content_parts)
            reply_text = response.text.strip()

            if not reply_text:
                await channel.send(
                    "🤐 โมเดลไม่ตอบอะไรเลย (อาจถูก Safety Filter บล็อก)",
                    reference=reference_msg,
                    mention_author=False,
                )
                return

            add_history(channel.id, "user", prompt)
            add_history(channel.id, "model", reply_text)
            await send_long_message(channel, reply_text, reference_msg)

        except genai.types.generation_types.BlockedPromptException as exc:
            print(f"🛑 Blocked: {exc}")
            await channel.send(
                "🚫 ข้อความถูกบล็อกโดย Safety Filter ลองเปลี่ยนคำถามนะครับ",
                reference=reference_msg,
                mention_author=False,
            )
        except Exception as exc:
            print(f"❌ AI Error: {type(exc).__name__}: {exc}")
            await channel.send(
                f"⚠️ เกิดข้อผิดพลาด: `{type(exc).__name__}`",
                reference=reference_msg,
                mention_author=False,
            )


# ===== EVENTS =====
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"🤖 Model: {MODEL_NAME} | History: {MAX_HISTORY} turns | Mode: {TRIGGER_MODE}")
    try:
        await tree.sync()
        print("🔧 Slash Commands Synced")
    except Exception as exc:
        print(f"⚠️ Sync failed: {exc}")
    print("------")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return

    await bot.process_commands(message)

    if TRIGGER_MODE == "slash":
        return

    is_mention = bot.user.mentioned_in(message)
    is_reply = (
        message.reference
        and message.reference.resolved
        and getattr(message.reference.resolved, "author", None) == bot.user
    )
    is_all_mode = TRIGGER_MODE == "all"

    if not (is_mention or is_reply or is_all_mode):
        return

    prompt = message.content
    if is_mention:
        prompt = prompt.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "")
    prompt = prompt.strip()

    if not prompt and not message.attachments:
        await message.reply("หละ? ถามอะไรมาไง? 🤔", mention_author=False)
        return

    await process_ai_response(
        message.channel,
        prompt or "อธิบายรูปนี้หน่อย",
        message,
        list(message.attachments),
    )


# ===== SLASH COMMANDS =====
@tree.command(name="ask", description="ถาม AI อะไรก็ได้")
@app_commands.describe(question="คำถามของคุณ", image="แนบรูปภาพมาให้ AI ดู (Optional)")
async def ask_slash(interaction: discord.Interaction, question: str, image: Optional[discord.Attachment] = None):
    await interaction.response.defer(thinking=True)
    await process_ai_response(
        interaction.channel,
        question,
        None,
        [image] if image else [],
    )
    # เมื่อ response ถูกส่งผ่าน channel แล้ว ไม่ต้องตอบซ้ำ


@tree.command(name="reset", description="ล้างประวัติการคุยในช่องนี้")
async def reset_slash(interaction: discord.Interaction):
    clear_history(interaction.channel.id)
    await interaction.response.send_message("🧠 ล้างความจำเรียบร้อย เริ่มคุยใหม่ได้เลย!", ephemeral=True)


@tree.command(name="config", description="ดูตั้งค่าบอท")
@app_commands.default_permissions(administrator=True)
async def config_slash(interaction: discord.Interaction):
    embed = discord.Embed(title="⚙️ Bot Config", color=0x00ff88)
    embed.add_field(name="Model", value=MODEL_NAME, inline=True)
    embed.add_field(name="Max History", value=f"{MAX_HISTORY} turns", inline=True)
    embed.add_field(name="Trigger Mode", value=TRIGGER_MODE, inline=True)
    embed.add_field(
        name="Allowed Channels",
        value=f"{len(ALLOWED_CHANNELS)} channels" if ALLOWED_CHANNELS else "All Channels",
        inline=False,
    )
    embed.add_field(name="System Prompt", value=(SYSTEM_PROMPT[:100] + "...") if len(SYSTEM_PROMPT) > 100 else SYSTEM_PROMPT, inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ===== PREFIX COMMANDS =====
@bot.command(name="reset", aliases=["clear", "forget"])
async def reset_prefix(ctx: commands.Context):
    clear_history(ctx.channel.id)
    await ctx.reply("🧠 ล้างความจำเรียบร้อย!", mention_author=False)


@bot.command(name="ask")
async def ask_prefix(ctx: commands.Context, *, question: str):
    await process_ai_response(ctx.channel, question, ctx.message, list(ctx.message.attachments))


# ===== RUN =====
if __name__ == "__main__":
    try:
        bot.run(DISCORD_TOKEN, log_handler=None)
    except discord.LoginFailure:
        print("❌ DISCORD_TOKEN ผิด หรือหมดอายุ")
    except Exception as exc:
        print(f"❌ Fatal: {exc}")
