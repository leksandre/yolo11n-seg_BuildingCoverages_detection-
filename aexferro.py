from some import TELEGRAM_BOT_TOKEN, service_chats_id, managers_chats_id, admin_chats_id, CERT_PATH, CLASS_NAMES_EN_TO_RU, CLASSES, MODEL_PATH


import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os




# === Логирование ===
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === Загрузка модели один раз при старте ===
logger.info("Загрузка модели сегментации...")
model = YOLO(MODEL_PATH)
logger.info("Модель успешно загружена.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Получаем фото в максимальном разрешении
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        file_bytes = await file.download_as_bytearray()

        # Конвертируем в OpenCV изображение
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            await update.message.reply_text("Не удалось загрузить изображение.")
            return

        # Инференс
        results = model(img, conf=0.3, imgsz=640)

        if not results or len(results[0].boxes) == 0:
            await update.message.reply_text("На изображении не обнаружено ни одного объекта.")
            return

        result = results[0]


        report = ""
        # --- Подсчёт площадей по маскам ---
        class_pixel_areas = {}
        total_mask_pixels = 0

        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data  # Tensor [N, H, W], dtype=torch.bool или uint8
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i, cls_id in enumerate(cls_ids):
                if cls_id >= len(CLASSES):
                    continue
                class_key = CLASSES[cls_id]
                mask = masks[i].cpu().numpy().astype(bool)
                area = mask.sum()  # количество пикселей
                class_pixel_areas[class_key] = class_pixel_areas.get(class_key, 0) + area
                total_mask_pixels += area

        # --- Формирование отчёта с процентами ---
        if total_mask_pixels > 0:
            report_lines = ["Обнаруженные поверхности (в % от общей покрытой площади):"]
            for class_key, area in sorted(class_pixel_areas.items(), key=lambda x: -x[1]):
                percent = (area / total_mask_pixels) * 100
                name_ru = CLASS_NAMES_EN_TO_RU.get(class_key, class_key)
                report_lines.append(f"– {name_ru}: {percent:.1f}%")
            report = "\n".join(report_lines)
        else:
            report = "Сегментированные области не обнаружены."

        # Подсчёт классов
        class_counts = {}
        if result.boxes and len(result.boxes.cls) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for cls_id in cls_ids:
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
                cls_key = CLASSES[cls_id]
                cls_name_ru = CLASS_NAMES_EN_TO_RU.get(cls_key, cls_key)  # fallback на оригинал при отсутствии
                class_counts[cls_name_ru] = class_counts.get(cls_name, 0) + 1

#         # Формируем текстовый отчёт
#         if class_counts:
#             report = report + "\n Обнаружены следующие поверхности:\n" + "\n".join(
#                 f"– {name}: {count}" for name, count in class_counts.items()
#             )
#         else:
#             report = report + "\n Объекты обнаружены, но не удалось определить классы."

        # Наложение масок и bbox на изображение
        plotted_img = result.plot()  # BGR numpy array
        plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(plotted_img_rgb)

        # Сохраняем в буфер
        bio = io.BytesIO()
        pil_image.save(bio, format="JPEG")
        bio.seek(0)

        # Отправляем фото + отчёт
        await update.message.reply_photo(photo=bio, caption=report)


        for chat_id_service in admin_chats_id:
            try:
                # 1. Отправляем исходное изображение
                await context.bot.send_photo(
                    chat_id=chat_id_service,
                    photo=update.message.photo[-1].file_id,  # оригинальное фото из Telegram
                    caption=f"📥 Исходное изображение от пользователя {update.effective_user.id} (@{update.effective_user.username})",
                )

                # 2. Отправляем обработанное изображение + отчёт
                bio.seek(0)  # сброс буфера перед повторной отправкой
                await context.bot.send_photo(
                    chat_id=chat_id_service,
                    photo=bio,
                    caption=f"🤖 Результат обработки:\n{report}",
                )
            except Exception as e:
                logger.error(f"Не удалось продублировать в служебный чат {chat_id_service}: {e}")

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения.")

# === Запуск бота ===
if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Обрабатываем ТОЛЬКО фото
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)
    application.add_handler(photo_handler)

    logger.info("Бот запущен и ожидает изображения...")
    application.run_polling()