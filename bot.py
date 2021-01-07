import os
import urllib

import telebot
import torch

import detection
import ocr

# Declare instance segmentation model
detection_model = detection.get_model(num_classes=2)
detection_model.load_state_dict(torch.load('./models/maskrcnn_resnet50_fpn.pth'))

# Declare OCR model
ocr_model = ocr.get_model()
ocr_model.load_state_dict(torch.load("./models/ocr_resnet18.pth"))

# Declare telegram bot
TOKEN = "your_token"
bot = telebot.TeleBot(TOKEN)

bot_text = """
Приветствуем вас в нашем демо боте!

Мы умеем находить и распознавать номера машин на фотографиях 🚀

Ждем от вас крутых фотографий 🤟

Создано с ❤️ нашей командой по ANPR.
"""


def save_image_from_message(message):
    cid = message.chat.id
    image_id = message.photo[len(message.photo) - 1].file_id
    bot.send_message(cid, "🔥 Анализирую, ждите... 🔥")
    file_path = bot.get_file(image_id).file_path
    image_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    image_name = f"{image_id}.jpg"
    urllib.request.urlretrieve(image_url, f"./tmp/{image_name}")
    return image_name


def cleanup_remove_image(image_name):
    folder = os.listdir("./detected_rois/")
    for image in folder:
        if image.endswith(".jpg"):
            os.remove(f"./detected_rois/{image}")
    os.remove(f"./tmp/{image_name}")


# Telegram bot
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.send_message(message.chat.id, bot_text)


@bot.message_handler(content_types=["photo"])
def handle(message):
    image_name = save_image_from_message(message)

    detection_pred_time = detection.predict(detection_model, "./tmp", image_name)

    folder = os.listdir("./detected_rois/")
    for image in folder:
        if image.endswith(".jpg"):
            bot.send_photo(message.chat.id,
                           open(f"./detected_rois/{image}", "rb"),
                           "Я что-то нашел...")

            predicted_number, ocr_pred_time = ocr.predict(ocr_model, "./detected_rois", image)

            output = "И распознал:\n" \
                     + f"<b>{predicted_number}</b>\n\n" \
                     + "Все это я выполнил за:\n" \
                     + f"<b><i>{detection_pred_time + ocr_pred_time:.4f} секунд</i></b>\n\n" \
                     + "🚀 Я хочу больше! 🚀"

            bot.reply_to(message, output, parse_mode='HTML')

    cleanup_remove_image(image_name)


if __name__ == "__main__":
    bot.polling()
