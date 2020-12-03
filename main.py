import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from fastai.vision.all import load_learner
import pickle
import os

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def start(update, context):
    update.message.reply_text(
        "Bot by @TejBhatt \n\n "
        "EN : Just send me a photo of you and I will tell you if you're wearing a mask 😏 \n"
        "HN : बस मुझे आपको एक फोटो भेजना है और अगर आप मास्क पहन रहे हैं तो मैं आपको बताऊंगा \n"
    )


def help_command(update, context):
    update.message.reply_text('My only purpose is to tell you if you are wearing a mask. Send a photo')


# def echo(update, context):
#     print(update)
#     print(context)
#     update.message.reply_text(update.message.text)


def load_model():
    global model
    #model = load_learner('model/model.pkl')
    model = load_learner('model3.pkl')
    print('Model loaded')


def detect_mask(update, context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')

    label = model.predict('user_photo.jpg')[0]
    if label == "with_mask":
        update.message.reply_text(
            "EN: Looks like you are wearing a mask 😷. I hope you don't forget it when going out!😉 \n\n"
            "HN: लगता है जैसे आपने मास्क पहना हो wearing। मुझे आशा है कि बाहर जाते समय आप इसे नहीं भूलेंगे!"
        )
    else:
        update.message.reply_text(
            "EN: Looks like you are not wearing a mask 😷. Please wear one and stay safe 🙄\n\n"
            "HN: लगता है कि आपने मास्क नहीं पहना है not कृपया एक पहनें और सुरक्षित रहें🙄"
        )


def main():
    load_model()
    updater = Updater(token="your_token", use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))

    dp.add_handler(MessageHandler(Filters.photo, detect_mask))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
