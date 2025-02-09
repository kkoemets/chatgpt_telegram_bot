import asyncio
import base64
import io
import logging
from datetime import datetime

import openai
import telegram
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import CallbackContext
from telegram.helpers import escape_markdown

import config
import database
import openai_utils
from constants import TELEGRAM_MESSAGE_MAX_LIMIT
from user import is_previous_message_not_answered_yet, register_user_if_not_exists, user_tasks, user_semaphores
from util import split_text_into_chunks

db = database.Database()
logger = logging.getLogger(__name__)


async def _vision_message_handle_fn(
        update: Update, context: CallbackContext, use_new_dialog_timeout: bool = True
):
    logger.info('_vision_message_handle_fn')
    user_id = update.message.from_user.id
    current_model = db.get_user_attribute(user_id, "current_model")

    if current_model != "gpt-4o":
        await update.message.reply_text(
            "🥲 Images processing is only available for <b>gpt-4o</b> model. Please change your settings in /settings",
            parse_mode=ParseMode.HTML,
        )
        return

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    # new dialog timeout
    if use_new_dialog_timeout:
        if (datetime.now() - db.get_user_attribute(user_id,
                                                   "last_interaction")).seconds > config.new_dialog_timeout and len(
            db.get_dialog_messages(user_id)) > 0:
            db.start_new_dialog(user_id)
            await update.message.reply_text(
                f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅",
                parse_mode=ParseMode.HTML)
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    buf = None
    if update.message.effective_attachment:
        photo = update.message.effective_attachment[-1]
        photo_file = await context.bot.get_file(photo.file_id)

        # store file in memory, not on disk
        buf = io.BytesIO()
        await photo_file.download_to_memory(buf)
        buf.name = "image.jpg"  # file extension is required
        buf.seek(0)  # move cursor to the beginning of the buffer

    # in case of CancelledError
    n_input_tokens, n_output_tokens = 0, 0

    try:
        # send placeholder message to user
        placeholder_message = await update.message.reply_text("...")
        message = update.message.caption or update.message.text or ''

        # send typing action
        await update.message.chat.send_action(action="typing")

        dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
        parse_mode = {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
            config.chat_modes[chat_mode]["parse_mode"]
        ]

        chatgpt_instance = openai_utils.ChatGPT(model=current_model,
                                                model_options=config.models["info"][current_model]['model_options'])
        gen = chatgpt_instance.send_vision_message_stream(
            message,
            dialog_messages=dialog_messages,
            image_buffer=buf,
            chat_mode=chat_mode,
        )

        prev_answer = ""
        async for gen_item in gen:
            (
                status,
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = gen_item

            answer = answer[:TELEGRAM_MESSAGE_MAX_LIMIT]  # telegram message limit

            # update only when 100 new symbols are ready
            if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                continue

            try:
                await context.bot.edit_message_text(
                    answer,
                    chat_id=placeholder_message.chat_id,
                    message_id=placeholder_message.message_id,
                    parse_mode=parse_mode,
                )
            except telegram.error.BadRequest as e:
                if str(e).startswith("Message is not modified"):
                    continue
                else:
                    await context.bot.edit_message_text(
                        answer,
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                    )

            await asyncio.sleep(0.01)  # wait a bit to avoid flooding

            prev_answer = answer

        # update user data
        if buf is not None:
            base_image = base64.b64encode(buf.getvalue()).decode("utf-8")
            new_dialog_message = {"user": [
                {
                    "type": "text",
                    "text": message,
                },
                {
                    "type": "image",
                    "image": base_image,
                }
            ]
                , "bot": answer, "date": datetime.now()}
        else:
            new_dialog_message = {"user": [{"type": "text", "text": message}], "bot": answer, "date": datetime.now()}

        db.set_dialog_messages(
            user_id,
            db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
            dialog_id=None
        )

        db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

    except asyncio.CancelledError:
        db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
        raise

    except Exception as e:
        error_text = f"Something went wrong during completion. Reason: {e}"
        logger.error(error_text)
        await update.message.reply_text(error_text)
        return


async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return

    current_model = db.get_user_attribute(user_id, "current_model")

    async def message_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id,
                                                       "last_interaction")).seconds > config.new_dialog_timeout and len(
                db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(
                    f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅",
                    parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")

            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                await update.message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!",
                                                parse_mode=ParseMode.HTML)
                return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model,
                                                    model_options=config.models["info"][current_model]['model_options'])
            gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages,
                                                       chat_mode=chat_mode)

            prev_answer = ""

            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                if len(answer) >= TELEGRAM_MESSAGE_MAX_LIMIT and status != "finished":
                    if (abs(len(answer) - len(prev_answer)) < 100):
                        continue

                    prev_answer = answer
                    await context.bot.edit_message_text(
                        f"Message length is too long, about {len(answer)} symbols already. Please wait.",
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                        parse_mode=parse_mode
                    )
                    continue

                if len(answer) >= TELEGRAM_MESSAGE_MAX_LIMIT:
                    chunks = split_text_into_chunks(answer, TELEGRAM_MESSAGE_MAX_LIMIT)

                    await context.bot.edit_message_text(
                        "Sending separate message chunks.",
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                        parse_mode=parse_mode
                    )

                    for chunk in chunks:
                        try:
                            await context.bot.send_message(
                            chat_id=placeholder_message.chat_id,
                            text=chunk,
                            parse_mode=parse_mode)
                        except BadRequest:
                            await context.bot.send_message(
                                chat_id=placeholder_message.chat_id,
                                text=chunk)
                else:
                    try:
                        await context.bot.edit_message_text(
                            answer,
                            chat_id=placeholder_message.chat_id,
                            message_id=placeholder_message.message_id,
                            parse_mode=parse_mode
                        )
                    except telegram.error.BadRequest as e:
                        if str(e).startswith("Message is not modified"):
                            continue
                        else:
                            await context.bot.edit_message_text(
                                answer,
                                chat_id=placeholder_message.chat_id,
                                message_id=placeholder_message.message_id
                            )

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding

                prev_answer = answer

            # update user data
            new_dialog_message = {"user": [{"type": "text", "text": _message}], "bot": answer, "date": datetime.now()}

            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

        except asyncio.CancelledError:
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        model_before_task_creation = current_model
        if update.message.photo is not None and len(
                update.message.photo) > 0:

            if current_model != "gpt-4o":
                current_model = "gpt-4o"
                db.set_user_attribute(user_id, "current_model", "gpt-4o")
            task = asyncio.create_task(
                _vision_message_handle_fn(update, context, use_new_dialog_timeout=use_new_dialog_timeout)
            )
        else:
            task = asyncio.create_task(
                message_handle_fn()
            )

        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            db.set_user_attribute(user_id, "current_model", model_before_task_creation)
            if user_id in user_tasks:
                del user_tasks[user_id]


async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)

    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds",
                          voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(update, context, message=transcribed_text)


async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        image_urls = await openai_utils.generate_images(message, n_images=config.return_n_generated_images,
                                                        size=config.image_size)
    except openai.InvalidRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # token usage
    db.set_user_attribute(user_id, "n_generated_images",
                          config.return_n_generated_images + db.get_user_attribute(user_id, "n_generated_images"))

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)


async def is_bot_mentioned(update: Update, context: CallbackContext):
    try:
        message = update.message

        if message.chat.type == "private":
            return True

        if message.text is not None and ("@" + context.bot.username) in message.text:
            return True

        if message.reply_to_message is not None:
            if message.reply_to_message.from_user.id == context.bot.id:
                return True
    except:
        return True
    else:
        return False


async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)
