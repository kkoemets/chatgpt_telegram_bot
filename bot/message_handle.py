import logging

from telegram import Update
from telegram.ext import CallbackContext

logger = logging.getLogger(__name__)


async def unsupport_message_handle(update: Update, context: CallbackContext, message=None):
    error_text = f"I don't know how to read files or videos. Send the picture in normal mode (Quick Mode)."
    logger.error(error_text)
    await update.message.reply_text(error_text)
    return
