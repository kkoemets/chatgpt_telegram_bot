
import base64
from io import BytesIO
import openai
from openai import AsyncOpenAI

import config
aclient = AsyncOpenAI(api_key=config.openai_api_key)
if config.openai_api_base is not None:
    openai.api_base = config.openai_api_base
import tiktoken
from openai import OpenAIError
import logging

logger = logging.getLogger(__name__)


class ChatGPT:
    def __init__(self, model="gpt-4o", model_options={}):
        self.model = model
        self.model_options = model_options

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                r = await aclient.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self.model_options
                )

                answer = r.choices[0].message.content
                answer = self._postprocess_answer(answer)

                n_input_tokens, n_output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
            except OpenAIError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                r_gen = await aclient.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **self.model_options
                )

                answer = ""
                async for r_item in r_gen:
                    delta = r_item.choices[0].delta
                    # Check if attribute exists, otherwise use get
                    if hasattr(delta, "content"):
                        content_piece = "" if delta.content is None else delta.content
                    else:
                        content_piece = delta.get("content", "")
                    if content_piece != "":
                        answer += content_piece
                        n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)

                        n_first_dialog_messages_removed = 0

                        yield "not_finished", answer, (
                            n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                answer = self._postprocess_answer(answer)
            except OpenAIError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (
            n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    async def send_vision_message(
            self,
            message,
            dialog_messages=[],
            chat_mode="assistant",
            image_buffer: BytesIO = None,
    ):
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model == "gpt-4o":
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, chat_mode, image_buffer
                    )
                    r = await aclient.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **self.model_options
                    )
                    answer = r.choices[0].message.content
                else:
                    raise ValueError(f"Unsupported model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = (
                    r.usage.prompt_tokens,
                    r.usage.completion_tokens,
                )
            except OpenAIError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(
            dialog_messages
        )

        return (
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
        )

    async def send_vision_message_stream(
            self,
            message,
            dialog_messages=[],
            chat_mode="assistant",
            image_buffer: BytesIO = None,
    ):
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model == "gpt-4o":
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, chat_mode, image_buffer
                    )

                    r_gen = await aclient.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **self.model_options
                    )
                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if hasattr(delta, "content"):
                            content_piece =  "" if delta.content is None else delta.content
                        else:
                            content_piece = delta.get("content", "")

                        if content_piece != "":
                            answer += content_piece
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                answer = self._postprocess_answer(answer)
            except OpenAIError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e
                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (
            n_input_tokens,
            n_output_tokens,
        ), n_first_dialog_messages_removed

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _encode_image(self, image_buffer: BytesIO) -> str:
        return base64.b64encode(image_buffer.read()).decode("utf-8")

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode, image_buffer: BytesIO = None):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]

        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})

        if image_buffer is not None:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image(image_buffer)}",
                        "detail": "high"
                    }}
                ]
            })
        else:
            messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="gpt-4o"):
        if model == "gpt-4o":
            encoding = tiktoken.encoding_for_model(model)
            tokens_per_message = 3
        elif model == "o3-mini":
            encoding = tiktoken.encoding_for_model("gpt-4o")
            tokens_per_message = 3
        else:
            encoding = tiktoken.encoding_for_model(model)
            tokens_per_message = 0
            print(f"Cannot count tokens for model {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            if isinstance(message["content"], list):
                for sub_message in message["content"]:
                    if "type" in sub_message:
                        if sub_message["type"] == "text":
                            n_input_tokens += len(encoding.encode(sub_message["text"]))
                        elif sub_message["type"] == "image_url":
                            pass
            else:
                if "type" in message:
                    if message["type"] == "text":
                        n_input_tokens += len(encoding.encode(message["text"]))
                    elif message["type"] == "image_url":
                        pass

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens


async def transcribe_audio(audio_file) -> str:
    r = await aclient.audio.transcribe("whisper-1", audio_file)
    return r.get("text", "") or ""


async def generate_images(prompt, n_images=4, size="512x512"):
    r = await aclient.images.generate(prompt=prompt, n=n_images, size=size)
    image_urls = [item.url for item in r.data]
    return image_urls


async def is_content_acceptable(prompt):
    r = await aclient.moderations.create(input=prompt)
    return not all(r.results[0].categories.values())
