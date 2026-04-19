"""запускает telegram-бота проекта:
принимает вопросы пользователя, отправляет их в rag-пайплайн,
формирует ответ с указанием источников и страниц,
и возвращает результат в telegram."""


from __future__ import annotations

import html
import logging
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
from src.rag.naive_rag.generation import llm_answer # pyright: ignore[reportMissingImports]
from src.rag.hybrid_rag.pipeline import ask_structured # pyright: ignore[reportMissingImports]
from src.settings import get_settings # pyright: ignore[reportMissingImports]


def _split_text(text: str, limit: int = 3900) -> list[str]:
    parts: list[str] = []
    rest = text or ""
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def _format_sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""

    lines = ["<b>📚 Источники</b>", ""]
    for s in sources:
        n = s.get("n")
        title = html.escape(str(s.get("title") or "Источник"))

        page_start = s.get("page_start")
        page_end = s.get("page_end")

        if page_start and page_end and page_start != page_end:
            page_text = f"стр. {page_start}-{page_end}"
        elif page_start:
            page_text = f"стр. {page_start}"
        else:
            page_text = "стр. n/a"

        lines.append(f"[{n}] {title} — {page_text}")

    return "\n".join(lines)

'''
def _format_urls_block(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["<b>Ссылки:</b>"]
    for s in sources:
        n = s.get("n")
        url = str(s.get("url") or "").strip()
        if not url:
            continue
        lines.append(f"[{n}] {html.escape(url)}")
    return "\n".join(lines)
'''

def _build_sources_keyboard(sources: list[dict]) -> InlineKeyboardMarkup | None:
    rows: list[list[InlineKeyboardButton]] = []

    for s in sources:
        url = str(s.get("url") or "").strip()
        if not _is_public_http_url(url):
            continue

        n = s.get("n")
        page_start = s.get("page_start")
        page_end = s.get("page_end")

        if page_start and page_end and page_start != page_end:
            page_text = f"{page_start}-{page_end}"
        elif page_start:
            page_text = f"{page_start}"
        else:
            page_text = "n/a"

        label = f"Открыть источник [{n}] — стр. {page_text}"
        rows.append([InlineKeyboardButton(text=label, url=url)])

    if not rows:
        return None

    return InlineKeyboardMarkup(inline_keyboard=rows)


def _is_public_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in {"http", "https"}:
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    return True


def _extract_ask_text(message_text: str, command_args: str | None) -> str:
    if command_args and command_args.strip():
        return command_args.strip()
    text = (message_text or "").strip()
    low = text.lower()
    if low.startswith("/ask"):
        rest = text[4:].strip()
        if rest.startswith("@"):
            pos = rest.find(" ")
            if pos != -1:
                rest = rest[pos + 1 :].strip()
            else:
                rest = ""
        return rest
    return ""


async def _answer_question(message: Message, question: str) -> None:
    s = get_settings()
    q = (question or "").strip()
    if not q:
        await message.answer("Отправьте вопрос текстом или используйте: /ask ваш вопрос")
        return

    wait_msg = await message.answer("🔎 Ищу ответ в источниках...")
    try:
        result = ask_structured(
            q,
            k=s.BOT_TOP_K,
            llm_answer=llm_answer,
        )
    except Exception:
        logging.exception("RAG ask failed")
        await wait_msg.edit_text("Ошибка при обработке запроса. Попробуйте позже.")
        return

    answer = result.get("answer", "В источниках нет ответа.")
    sources = result.get("sources", [])

    body = f"<b>🤖 Ответ</b>\n\n{html.escape(answer)}"
    chunks = _split_text(body)
    if not chunks:
        chunks = ["В источниках нет ответа."]

    await wait_msg.edit_text(chunks[0], parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    for chunk in chunks[1:]:
        await message.answer(chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    if sources:
        sources_block = _format_sources_html(sources)
        kb = _build_sources_keyboard(sources)
        await message.answer(
            sources_block,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=kb,
        )
'''
        urls_block = _format_urls_block(sources)
        if urls_block:
            await message.answer(
                urls_block,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
'''
            
def create_dispatcher() -> Dispatcher:
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def on_start(message: Message) -> None:
        text = (
            "Привет! Я Aroma-bot 🌿 для поиска ответов по библиотеке книг про ЭФИРНЫЕ МАСЛА 🧴\n\n"
            "Что я умею:\n"
            "— ищу информацию только по загруженным источникам\n"
            "— показываю, на какой странице найден ответ\n"
            "— даю ссылки на источник\n\n"
            "Просто напишите вопрос. Например: Что такое ароматерапия ?"
        )
        await message.answer(text)

    @dp.message(Command("help"))
    async def on_help(message: Message) -> None:
        text = (
            "Как пользоваться ботом:\n\n"
            "1) Напишите вопрос обычным сообщением\n"
            "2) Перейди по ссылке и прочитай полный ответ в источнике"
            "Бот отвечает только по загруженным источникам и показывает страницу, где найдена информация."
        )
        await message.answer(text)

    @dp.message(Command("ask"))
    async def on_ask(message: Message, command: CommandObject) -> None:
        q = _extract_ask_text(message.text or "", command.args)
        await _answer_question(message, q)

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        await _answer_question(message, message.text or "")

    return dp


async def run_bot() -> None:
    s = get_settings()
    if not s.TELEGRAM_BOT_TOKEN.strip():
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty in .env")

    bot = Bot(token=s.TELEGRAM_BOT_TOKEN)
    dp = create_dispatcher()
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_bot())
